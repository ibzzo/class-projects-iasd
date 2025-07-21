#!/usr/bin/env python3
"""
ULTIMATE OPTUNA FULL OPTIMIZATION - Version complète pour AUC > 0.92
Recherche exhaustive des meilleurs hyperparamètres
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy import stats
import pywt
from hmmlearn import hmm
from collections import Counter
import optuna
from optuna.samplers import TPESampler
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

print("="*80)
print("ULTIMATE OPTUNA FULL OPTIMIZATION")
print("Version complète avec recherche exhaustive")
print("="*80)

# Configuration globale
RANDOM_STATE = 42
N_FOLDS = 5
N_TRIALS = 200  # Augmenter pour une recherche plus approfondie

# 1. CHARGEMENT DES DONNÉES
print("\n[1] Chargement des données...")
train = pd.read_csv('train.csv', parse_dates=['Date'])
test = pd.read_csv('test.csv', parse_dates=['Date'])
test_dates = test['Date'].copy()

train_dist = train['Market_Regime'].value_counts(normalize=True).sort_index()
print("\nDistribution cible:")
for regime in [-1, 0, 1]:
    print(f"  Régime {regime}: {train_dist[regime]*100:.1f}%")

# Variables globales pour l'optimisation
global_train = None
global_test = None
global_y = None
global_price_cols = None

# 2. FEATURE ENGINEERING COMPLET
def create_advanced_features(df, params):
    """Feature engineering avec paramètres ajustables"""
    price_cols = [col for col in df.columns if col not in ['Date', 'Market_Regime']]
    
    # 1. RETURNS ET RATIOS
    for col in price_cols[:params['n_price_cols']]:
        # Returns multiples
        for period in params['return_periods']:
            df[f'{col}_ret_{period}'] = df[col].pct_change(period)
            
        # Ratios de returns
        if params['use_return_ratios']:
            df[f'{col}_ret_ratio_5_20'] = df[f'{col}_ret_5'] / (df[f'{col}_ret_20'] + 1e-8)
            df[f'{col}_ret_ratio_10_50'] = df[f'{col}_ret_10'] / (df[f'{col}_ret_50'] + 1e-8) if 50 in params['return_periods'] else 0
        
    # 2. VOLATILITÉ
    for col in price_cols[:params['n_vol_cols']]:
        returns = df[col].pct_change()
        
        # Volatilité rolling
        for window in params['vol_windows']:
            df[f'{col}_vol_{window}'] = returns.rolling(window).std() * np.sqrt(252)
        
        # EWMA volatility
        if params['use_ewma_vol']:
            for span in params['ewma_spans']:
                df[f'{col}_ewma_vol_{span}'] = returns.ewm(span=span).std() * np.sqrt(252)
        
        # Volatility ratios
        if params['use_vol_ratios']:
            df[f'{col}_vol_ratio'] = df[f'{col}_vol_{params["vol_windows"][0]}'] / (df[f'{col}_vol_{params["vol_windows"][-1]}'] + 1e-8)
    
    # 3. WAVELETS
    if params['use_wavelets']:
        for col in price_cols[:params['n_wavelet_cols']]:
            try:
                for window in params['wavelet_windows']:
                    for wavelet_type in params['wavelet_types']:
                        df[f'{col}_wavelet_{wavelet_type}_{window}'] = df[col].rolling(window).apply(
                            lambda x: np.std(pywt.dwt(x, wavelet_type)[0]) if len(x) == window else np.nan
                        )
            except:
                pass
    
    # 4. MICROSTRUCTURE
    if params['use_microstructure']:
        for col in price_cols[:params['n_micro_cols']]:
            returns = df[col].pct_change()
            
            # Realized volatility
            df[f'{col}_realized_vol'] = returns.rolling(params['micro_window']).apply(
                lambda x: np.sqrt(np.sum(x**2))
            )
            
            # Sign changes
            df[f'{col}_sign_changes'] = returns.rolling(params['micro_window']).apply(
                lambda x: np.sum(np.diff(np.sign(x)) != 0)
            )
            
            # Autocorrelation
            if params['use_autocorr']:
                for lag in params['autocorr_lags']:
                    df[f'{col}_autocorr_{lag}'] = returns.rolling(params['micro_window']).apply(
                        lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
                    )
    
    # 5. CORRÉLATIONS
    if params['use_correlations']:
        for window in params['corr_windows']:
            corr_features = []
            for i in range(len(df)):
                if i >= window:
                    corr_matrix = df[price_cols[:params['n_corr_cols']]].iloc[i-window:i].corr()
                    upper_tri = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                    avg_corr = np.nanmean(upper_tri)
                    std_corr = np.nanstd(upper_tri)
                    try:
                        eigenvalues = np.linalg.eigvals(corr_matrix)
                        max_eigen = np.max(eigenvalues)
                        eigen_ratio = max_eigen / np.sum(eigenvalues)
                    except:
                        max_eigen = np.nan
                        eigen_ratio = np.nan
                    corr_features.append([avg_corr, std_corr, max_eigen, eigen_ratio])
                else:
                    corr_features.append([np.nan, np.nan, np.nan, np.nan])
            
            corr_df = pd.DataFrame(corr_features, 
                                  columns=[f'avg_corr_{window}', f'std_corr_{window}', 
                                         f'max_eigen_{window}', f'eigen_ratio_{window}'])
            for col in corr_df.columns:
                df[col] = corr_df[col]
    
    # 6. INDICATEURS TECHNIQUES
    for col in price_cols[:params['n_tech_cols']]:
        # RSI
        delta = df[col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(params['rsi_period']).mean()
        rs = gain / (loss + 1e-8)
        df[f'{col}_rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        for bb_window in params['bb_windows']:
            sma = df[col].rolling(bb_window).mean()
            std = df[col].rolling(bb_window).std()
            df[f'{col}_bb_upper_{bb_window}'] = (df[col] - (sma + 2*std)) / (df[col] + 1e-8)
            df[f'{col}_bb_lower_{bb_window}'] = ((sma - 2*std) - df[col]) / (df[col] + 1e-8)
            df[f'{col}_bb_width_{bb_window}'] = (2 * 2 * std) / (sma + 1e-8)
        
        # MACD
        if params['use_macd']:
            ema_fast = df[col].ewm(span=params['macd_fast']).mean()
            ema_slow = df[col].ewm(span=params['macd_slow']).mean()
            df[f'{col}_macd'] = ema_fast - ema_slow
            df[f'{col}_macd_signal'] = df[f'{col}_macd'].ewm(span=params['macd_signal']).mean()
            df[f'{col}_macd_diff'] = df[f'{col}_macd'] - df[f'{col}_macd_signal']
    
    # 7. TURBULENCE INDEX
    if params['use_turbulence']:
        returns_matrix = df[price_cols[:params['n_turb_cols']]].pct_change()
        turbulence = []
        turbulence_pca = []
        
        for i in range(len(df)):
            if i >= params['turb_lookback']:
                historical = returns_matrix.iloc[i-params['turb_lookback']:i]
                current = returns_matrix.iloc[i]
                try:
                    # Turbulence classique
                    mean_hist = historical.mean()
                    cov_hist = historical.cov()
                    inv_cov = np.linalg.pinv(cov_hist)
                    diff = current - mean_hist
                    turb = np.sqrt(diff @ inv_cov @ diff)
                    turbulence.append(turb)
                    
                    # Turbulence PCA
                    if params['use_turb_pca']:
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=min(5, len(price_cols[:params['n_turb_cols']])))
                        pca.fit(historical)
                        current_pca = pca.transform(current.values.reshape(1, -1))
                        historical_pca = pca.transform(historical)
                        mean_pca = historical_pca.mean(axis=0)
                        cov_pca = np.cov(historical_pca.T)
                        inv_cov_pca = np.linalg.pinv(cov_pca)
                        diff_pca = current_pca[0] - mean_pca
                        turb_pca = np.sqrt(diff_pca @ inv_cov_pca @ diff_pca)
                        turbulence_pca.append(turb_pca)
                    else:
                        turbulence_pca.append(np.nan)
                except:
                    turbulence.append(np.nan)
                    turbulence_pca.append(np.nan)
            else:
                turbulence.append(np.nan)
                turbulence_pca.append(np.nan)
        
        df['turbulence_index'] = turbulence
        if params['use_turb_pca']:
            df['turbulence_pca'] = turbulence_pca
    
    # 8. FEATURES TEMPORELLES
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_month'] = df['Date'].dt.day
    df['week_of_year'] = df['Date'].dt.isocalendar().week
    
    # Encodage cyclique
    if params['use_cyclical_encoding']:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['dom_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['dom_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
    
    # Indicateurs spéciaux
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_quarter_end'] = df['Date'].dt.is_quarter_end.astype(int)
    df['is_quarter_start'] = df['Date'].dt.is_quarter_start.astype(int)
    
    # 9. FEATURES STATISTIQUES AVANCÉES
    if params['use_advanced_stats']:
        for col in price_cols[:params['n_stats_cols']]:
            returns = df[col].pct_change()
            
            # Moments statistiques
            for window in params['stats_windows']:
                df[f'{col}_skew_{window}'] = returns.rolling(window).skew()
                df[f'{col}_kurt_{window}'] = returns.rolling(window).kurt()
                
                # Quantiles
                df[f'{col}_q25_{window}'] = returns.rolling(window).quantile(0.25)
                df[f'{col}_q75_{window}'] = returns.rolling(window).quantile(0.75)
                df[f'{col}_iqr_{window}'] = df[f'{col}_q75_{window}'] - df[f'{col}_q25_{window}']
    
    return df

# 3. HMM FEATURES
def add_hmm_features(train_df, test_df, price_cols, params):
    """HMM features avec paramètres ajustables"""
    # Préparer les données
    train_returns = train_df[price_cols[:params['hmm_n_cols']]].pct_change().fillna(0)
    train_returns = train_returns.replace([np.inf, -np.inf], 0)
    
    test_returns = test_df[price_cols[:params['hmm_n_cols']]].pct_change().fillna(0)
    test_returns = test_returns.replace([np.inf, -np.inf], 0)
    
    # HMM
    try:
        hmm_model = hmm.GaussianHMM(
            n_components=params['hmm_n_states'], 
            covariance_type=params['hmm_covariance_type'],
            n_iter=params['hmm_n_iter'],
            random_state=RANDOM_STATE
        )
        hmm_model.fit(train_returns[1:])
        
        # Prédictions
        train_proba = hmm_model.predict_proba(train_returns[1:])
        test_proba = hmm_model.predict_proba(test_returns)
        
        # Ajouter aux dataframes
        for i in range(params['hmm_n_states']):
            train_df[f'hmm_state_{i}'] = np.concatenate([[0], train_proba[:, i]])
            test_df[f'hmm_state_{i}'] = test_proba[:, i]
        
        # Score de log-vraisemblance
        if params['use_hmm_score']:
            train_scores = hmm_model.score_samples(train_returns[1:])[0]
            test_scores = hmm_model.score_samples(test_returns)[0]
            train_df['hmm_score'] = np.concatenate([[0], train_scores])
            test_df['hmm_score'] = test_scores
            
    except Exception as e:
        print(f"  Erreur HMM: {e}")
        for i in range(params['hmm_n_states']):
            train_df[f'hmm_state_{i}'] = 0
            test_df[f'hmm_state_{i}'] = 0
        if params['use_hmm_score']:
            train_df['hmm_score'] = 0
            test_df['hmm_score'] = 0
    
    return train_df, test_df

# 4. FONCTION OBJECTIVE OPTUNA
def objective(trial):
    """Fonction objective pour Optuna avec exploration complète"""
    
    # Hyperparamètres pour le feature engineering
    fe_params = {
        # Colonnes à utiliser
        'n_price_cols': trial.suggest_int('n_price_cols', 15, 25),
        'n_vol_cols': trial.suggest_int('n_vol_cols', 10, 20),
        'n_wavelet_cols': trial.suggest_int('n_wavelet_cols', 5, 15),
        'n_micro_cols': trial.suggest_int('n_micro_cols', 5, 15),
        'n_corr_cols': trial.suggest_int('n_corr_cols', 8, 15),
        'n_tech_cols': trial.suggest_int('n_tech_cols', 8, 15),
        'n_turb_cols': trial.suggest_int('n_turb_cols', 8, 15),
        'n_stats_cols': trial.suggest_int('n_stats_cols', 5, 10),
        
        # Périodes returns
        'return_periods': sorted([
            trial.suggest_int('ret_period_1', 1, 3),
            trial.suggest_int('ret_period_2', 5, 10),
            trial.suggest_int('ret_period_3', 15, 30),
            trial.suggest_int('ret_period_4', 40, 60)
        ]),
        'use_return_ratios': trial.suggest_categorical('use_return_ratios', [True, False]),
        
        # Volatilité
        'vol_windows': sorted([
            trial.suggest_int('vol_window_1', 5, 10),
            trial.suggest_int('vol_window_2', 15, 30),
            trial.suggest_int('vol_window_3', 40, 60)
        ]),
        'use_ewma_vol': trial.suggest_categorical('use_ewma_vol', [True, False]),
        'ewma_spans': [10, 20, 30] if trial.params.get('use_ewma_vol', False) else [],
        'use_vol_ratios': trial.suggest_categorical('use_vol_ratios', [True, False]),
        
        # Wavelets
        'use_wavelets': trial.suggest_categorical('use_wavelets', [True, False]),
        'wavelet_windows': [32, 64] if trial.params.get('use_wavelets', False) else [],
        'wavelet_types': ['db4', 'sym5'] if trial.params.get('use_wavelets', False) else [],
        
        # Microstructure
        'use_microstructure': trial.suggest_categorical('use_microstructure', [True, False]),
        'micro_window': trial.suggest_int('micro_window', 15, 30),
        'use_autocorr': trial.suggest_categorical('use_autocorr', [True, False]),
        'autocorr_lags': [1, 2, 5] if trial.params.get('use_autocorr', False) else [],
        
        # Corrélations
        'use_correlations': trial.suggest_categorical('use_correlations', [True, False]),
        'corr_windows': [20, 40] if trial.params.get('use_correlations', False) else [],
        
        # Indicateurs techniques
        'rsi_period': trial.suggest_int('rsi_period', 10, 20),
        'bb_windows': [20, 30],
        'use_macd': trial.suggest_categorical('use_macd', [True, False]),
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        
        # Turbulence
        'use_turbulence': trial.suggest_categorical('use_turbulence', [True, False]),
        'turb_lookback': trial.suggest_int('turb_lookback', 50, 100),
        'use_turb_pca': trial.suggest_categorical('use_turb_pca', [True, False]),
        
        # Temporel
        'use_cyclical_encoding': trial.suggest_categorical('use_cyclical_encoding', [True, False]),
        
        # Stats avancées
        'use_advanced_stats': trial.suggest_categorical('use_advanced_stats', [True, False]),
        'stats_windows': [20, 50] if trial.params.get('use_advanced_stats', False) else [],
        
        # HMM
        'hmm_n_states': trial.suggest_int('hmm_n_states', 3, 7),
        'hmm_n_cols': trial.suggest_int('hmm_n_cols', 10, 20),
        'hmm_covariance_type': trial.suggest_categorical('hmm_covariance_type', ['diag', 'full']),
        'hmm_n_iter': trial.suggest_int('hmm_n_iter', 50, 200),
        'use_hmm_score': trial.suggest_categorical('use_hmm_score', [True, False]),
    }
    
    # Hyperparamètres des modèles
    model_params = {
        # XGBoost
        'xgb_n_estimators': trial.suggest_int('xgb_n_estimators', 300, 1000),
        'xgb_max_depth': trial.suggest_int('xgb_max_depth', 4, 12),
        'xgb_learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.15, log=True),
        'xgb_subsample': trial.suggest_float('xgb_subsample', 0.6, 0.95),
        'xgb_colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 0.95),
        'xgb_gamma': trial.suggest_float('xgb_gamma', 0.0, 1.0),
        'xgb_reg_alpha': trial.suggest_float('xgb_reg_alpha', 0.0, 5.0),
        'xgb_reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.0, 5.0),
        'xgb_min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
        
        # LightGBM
        'lgb_n_estimators': trial.suggest_int('lgb_n_estimators', 300, 1000),
        'lgb_num_leaves': trial.suggest_int('lgb_num_leaves', 20, 80),
        'lgb_learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.15, log=True),
        'lgb_feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.5, 0.95),
        'lgb_bagging_fraction': trial.suggest_float('lgb_bagging_fraction', 0.5, 0.95),
        'lgb_bagging_freq': trial.suggest_int('lgb_bagging_freq', 1, 10),
        'lgb_min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 50),
        'lgb_lambda_l1': trial.suggest_float('lgb_lambda_l1', 0.0, 5.0),
        'lgb_lambda_l2': trial.suggest_float('lgb_lambda_l2', 0.0, 5.0),
        
        # CatBoost
        'cat_iterations': trial.suggest_int('cat_iterations', 300, 1000),
        'cat_depth': trial.suggest_int('cat_depth', 4, 10),
        'cat_learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.15, log=True),
        'cat_l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1.0, 20.0),
        'cat_border_count': trial.suggest_int('cat_border_count', 32, 255),
        
        # Extra Trees
        'extra_n_estimators': trial.suggest_int('extra_n_estimators', 200, 800),
        'extra_max_depth': trial.suggest_int('extra_max_depth', 10, 30),
        'extra_min_samples_split': trial.suggest_int('extra_min_samples_split', 5, 30),
        'extra_min_samples_leaf': trial.suggest_int('extra_min_samples_leaf', 2, 15),
        'extra_max_features': trial.suggest_categorical('extra_max_features', ['sqrt', 'log2', None]),
        
        # Sélection de features
        'n_features': trial.suggest_int('n_features', 150, 400),
        'feature_selection_method': trial.suggest_categorical('feature_selection_method', ['rf', 'mutual_info', 'both']),
        
        # Ensemble
        'ensemble_method': trial.suggest_categorical('ensemble_method', ['weighted', 'blending', 'stacking']),
        
        # Post-processing
        'use_calibration': trial.suggest_categorical('use_calibration', [True, False]),
        'adjustment_strength': trial.suggest_float('adjustment_strength', 0.1, 0.6),
        'persistence_factor': trial.suggest_float('persistence_factor', 1.0, 1.5),
        'turbulence_threshold': trial.suggest_float('turbulence_threshold', 0.75, 0.95),
    }
    
    # Créer les features
    print(f"\nTrial {trial.number}: Création des features...")
    train_features = create_advanced_features(global_train.copy(), fe_params)
    test_features = create_advanced_features(global_test.copy(), fe_params)
    
    # Ajouter HMM
    train_features, test_features = add_hmm_features(train_features, test_features, global_price_cols, fe_params)
    
    # Préparer les données
    feature_cols = [col for col in train_features.columns if col not in ['Date', 'Market_Regime']]
    X = train_features[feature_cols]
    X_test = test_features[feature_cols]
    
    # Nettoyer
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Supprimer colonnes avec trop de NaN
    valid_cols = X.columns[X.isnull().sum() < len(X) * 0.5]
    X = X[valid_cols]
    X_test = X_test[valid_cols]
    
    # Supprimer premières lignes
    min_valid_row = 60
    X = X.iloc[min_valid_row:].reset_index(drop=True)
    y = global_y[min_valid_row:]
    
    # Sélection de features
    if model_params['feature_selection_method'] in ['rf', 'both']:
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X, y)
        rf_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
    
    if model_params['feature_selection_method'] in ['mutual_info', 'both']:
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
        mi_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
    
    # Combiner les méthodes
    if model_params['feature_selection_method'] == 'both':
        combined_importances = pd.merge(
            rf_importances, mi_importances, 
            on='feature', suffixes=('_rf', '_mi')
        )
        combined_importances['importance'] = (
            combined_importances['importance_rf'] * 0.7 + 
            combined_importances['importance_mi'] * 0.3
        )
        feature_importances = combined_importances[['feature', 'importance']].sort_values('importance', ascending=False)
    elif model_params['feature_selection_method'] == 'rf':
        feature_importances = rf_importances
    else:
        feature_importances = mi_importances
    
    # Sélectionner top features
    n_features = min(model_params['n_features'], len(feature_importances))
    top_features = feature_importances.head(n_features)['feature'].tolist()
    
    # Toujours inclure features clés
    key_features = [col for col in X.columns if any(x in col for x in ['hmm_', 'turbulence', 'avg_corr', 'wavelet', 'eigen'])]
    selected_features = list(set(top_features + key_features))[:model_params['n_features']]
    
    X_selected = X[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Normalisation
    scaler = RobustScaler()
    X_selected_scaled = scaler.fit_transform(X_selected)
    X_test_selected_scaled = scaler.transform(X_test_selected)
    
    # Créer les modèles
    models = {
        'xgb': xgb.XGBClassifier(
            n_estimators=model_params['xgb_n_estimators'],
            max_depth=model_params['xgb_max_depth'],
            learning_rate=model_params['xgb_learning_rate'],
            subsample=model_params['xgb_subsample'],
            colsample_bytree=model_params['xgb_colsample_bytree'],
            gamma=model_params['xgb_gamma'],
            reg_alpha=model_params['xgb_reg_alpha'],
            reg_lambda=model_params['xgb_reg_lambda'],
            min_child_weight=model_params['xgb_min_child_weight'],
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric='mlogloss',
            use_label_encoder=False
        ),
        
        'lgb': lgb.LGBMClassifier(
            n_estimators=model_params['lgb_n_estimators'],
            num_leaves=model_params['lgb_num_leaves'],
            learning_rate=model_params['lgb_learning_rate'],
            feature_fraction=model_params['lgb_feature_fraction'],
            bagging_fraction=model_params['lgb_bagging_fraction'],
            bagging_freq=model_params['lgb_bagging_freq'],
            min_child_samples=model_params['lgb_min_child_samples'],
            lambda_l1=model_params['lgb_lambda_l1'],
            lambda_l2=model_params['lgb_lambda_l2'],
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1
        ),
        
        'cat': CatBoostClassifier(
            iterations=model_params['cat_iterations'],
            learning_rate=model_params['cat_learning_rate'],
            depth=model_params['cat_depth'],
            l2_leaf_reg=model_params['cat_l2_leaf_reg'],
            border_count=model_params['cat_border_count'],
            random_state=RANDOM_STATE,
            verbose=False,
            thread_count=-1
        ),
        
        'extra': ExtraTreesClassifier(
            n_estimators=model_params['extra_n_estimators'],
            max_depth=model_params['extra_max_depth'],
            min_samples_split=model_params['extra_min_samples_split'],
            min_samples_leaf=model_params['extra_min_samples_leaf'],
            max_features=model_params['extra_max_features'],
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    
    for name, model in models.items():
        model_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected_scaled, y)):
            X_train, X_val = X_selected_scaled[train_idx], X_selected_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Entraîner
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_train, y_train)
            
            # Prédire
            val_pred = model_clone.predict_proba(X_val)
            
            # Score
            score = roc_auc_score(y_val, val_pred, multi_class='ovr')
            model_scores.append(score)
        
        cv_scores.extend(model_scores)
    
    # Score moyen
    mean_score = np.mean(cv_scores)
    
    print(f"Trial {trial.number} - Score: {mean_score:.4f}")
    
    return mean_score

# 5. FONCTION PRINCIPALE
def main():
    global global_train, global_test, global_y, global_price_cols
    
    # Préparer les données globales
    global_train = train.copy()
    global_test = test.copy()
    global_price_cols = [col for col in train.columns if col not in ['Date', 'Market_Regime']]
    
    # Labels
    label_encoder = LabelEncoder()
    global_y = label_encoder.fit_transform(train['Market_Regime'])
    
    # Créer l'étude Optuna
    print("\n[2] Création de l'étude Optuna...")
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_STATE),
        study_name='ultimate_auc_optimization'
    )
    
    # Callbacks pour sauvegarder régulièrement
    def save_callback(study, trial):
        if trial.number % 10 == 0:
            # Sauvegarder l'étude
            with open(f'optuna_study_trial_{trial.number}.pkl', 'wb') as f:
                import pickle
                pickle.dump(study, f)
            
            # Sauvegarder les meilleurs params
            with open(f'best_params_trial_{trial.number}.json', 'w') as f:
                json.dump(study.best_params, f, indent=2)
    
    # Optimiser
    print(f"\n[3] Lancement de l'optimisation ({N_TRIALS} trials)...")
    print("Cela peut prendre plusieurs heures...")
    
    start_time = datetime.now()
    
    study.optimize(
        objective, 
        n_trials=N_TRIALS,
        callbacks=[save_callback],
        show_progress_bar=True
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Résultats
    print("\n" + "="*80)
    print("OPTIMISATION TERMINÉE!")
    print(f"Durée totale: {duration}")
    print(f"Nombre de trials: {len(study.trials)}")
    print(f"Meilleur score: {study.best_value:.4f}")
    print("\nMeilleurs hyperparamètres:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    # Sauvegarder les résultats finaux
    with open('final_best_params.json', 'w') as f:
        json.dump({
            'best_score': study.best_value,
            'best_params': study.best_params,
            'n_trials': N_TRIALS,
            'duration': str(duration)
        }, f, indent=2)
    
    # Visualisations Optuna
    print("\n[4] Génération des visualisations...")
    
    try:
        import plotly
        from optuna.visualization import plot_optimization_history, plot_param_importances
        
        # Historique d'optimisation
        fig = plot_optimization_history(study)
        fig.write_html('optimization_history.html')
        
        # Importance des paramètres
        fig = plot_param_importances(study)
        fig.write_html('param_importances.html')
        
        print("Visualisations sauvegardées:")
        print("  - optimization_history.html")
        print("  - param_importances.html")
    except:
        print("Impossible de générer les visualisations (plotly requis)")
    
    # Entraînement final avec les meilleurs paramètres
    print("\n[5] Entraînement final avec les meilleurs paramètres...")
    
    # TODO: Implémenter l'entraînement final et la génération de submission
    # Ce code sera identique à la partie finale de ultimate_optuna_optimized.py
    # mais avec les meilleurs paramètres trouvés
    
    print("\n" + "="*80)
    print("PROCESSUS COMPLET TERMINÉ!")
    print("Prochaine étape: Utiliser les paramètres dans final_best_params.json")
    print("pour entraîner le modèle final et générer la submission")
    print("="*80)

if __name__ == "__main__":
    main()