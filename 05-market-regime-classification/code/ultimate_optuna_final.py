#!/usr/bin/env python3
"""
ULTIMATE OPTUNA FINAL - Optimisation complète du modèle 0.9121
==============================================================
Version finale avec recherche exhaustive des hyperparamètres
Basé sur ultimate_robust_model.py (AUC 0.9121)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
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
import pickle
from datetime import datetime
import gc
warnings.filterwarnings('ignore')

print("="*80)
print("ULTIMATE OPTUNA FINAL - Optimisation du modèle 0.9121")
print("="*80)

# Configuration
RANDOM_STATE = 42
N_FOLDS = 5
N_TRIALS = 300  # Augmenté pour recherche exhaustive

# 1. CHARGEMENT DES DONNÉES
print("\n[1] Chargement des données...")
train = pd.read_csv('train.csv', parse_dates=['Date'])
test = pd.read_csv('test.csv', parse_dates=['Date'])
test_dates = test['Date'].copy()

train_dist = train['Market_Regime'].value_counts(normalize=True).sort_index()
print("\nDistribution cible:")
for regime in [-1, 0, 1]:
    print(f"  Régime {regime}: {train_dist[regime]*100:.1f}%")

# 2. FEATURE ENGINEERING COMPLET (du modèle original)
def create_advanced_features(df):
    """Features exactes du modèle à 0.9121"""
    price_cols = [col for col in df.columns if col not in ['Date', 'Market_Regime']]
    
    # 1. RETURNS ET RATIOS
    for col in price_cols[:20]:
        for period in [1, 2, 5, 10, 20]:
            df[f'{col}_ret_{period}'] = df[col].pct_change(period)
        df[f'{col}_ret_ratio_5_20'] = df[f'{col}_ret_5'] / (df[f'{col}_ret_20'] + 1e-8)
        
    # 2. VOLATILITÉ AVANCÉE
    for col in price_cols[:15]:
        returns = df[col].pct_change()
        for window in [5, 10, 20, 30]:
            df[f'{col}_vol_{window}'] = returns.rolling(window).std() * np.sqrt(252)
        df[f'{col}_ewma_vol'] = returns.ewm(span=20).std() * np.sqrt(252)
        df[f'{col}_vol_ratio'] = df[f'{col}_vol_5'] / (df[f'{col}_vol_20'] + 1e-8)
        
    # 3. WAVELETS
    for col in price_cols[:10]:
        try:
            for window in [32, 64]:
                df[f'{col}_wavelet_{window}'] = df[col].rolling(window).apply(
                    lambda x: np.std(pywt.dwt(x, 'db4')[0]) if len(x) == window else np.nan
                )
        except:
            pass
    
    # 4. MICROSTRUCTURE
    for col in price_cols[:10]:
        returns = df[col].pct_change()
        df[f'{col}_realized_vol'] = returns.rolling(20).apply(
            lambda x: np.sqrt(np.sum(x**2))
        )
        df[f'{col}_sign_changes'] = returns.rolling(20).apply(
            lambda x: np.sum(np.diff(np.sign(x)) != 0)
        )
        
    # 5. CORRÉLATIONS
    for window in [20, 30]:
        corr_features = []
        for i in range(len(df)):
            if i >= window:
                corr_matrix = df[price_cols[:10]].iloc[i-window:i].corr()
                upper_tri = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                avg_corr = np.nanmean(upper_tri)
                try:
                    max_eigen = np.max(np.linalg.eigvals(corr_matrix))
                except:
                    max_eigen = np.nan
                corr_features.append([avg_corr, max_eigen])
            else:
                corr_features.append([np.nan, np.nan])
        
        corr_df = pd.DataFrame(corr_features, columns=[f'avg_corr_{window}', f'max_eigen_{window}'])
        df[f'avg_corr_{window}'] = corr_df[f'avg_corr_{window}']
        df[f'max_eigen_{window}'] = corr_df[f'max_eigen_{window}']
    
    # 6. INDICATEURS TECHNIQUES
    for col in price_cols[:10]:
        # RSI
        delta = df[col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df[f'{col}_rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma = df[col].rolling(20).mean()
        std = df[col].rolling(20).std()
        df[f'{col}_bb_upper'] = (df[col] - (sma + 2*std)) / df[col]
        df[f'{col}_bb_lower'] = ((sma - 2*std) - df[col]) / df[col]
        
    # 7. TURBULENCE INDEX
    returns_matrix = df[price_cols[:10]].pct_change()
    turbulence = []
    for i in range(len(df)):
        if i >= 60:
            historical = returns_matrix.iloc[i-60:i]
            current = returns_matrix.iloc[i]
            try:
                mean_hist = historical.mean()
                cov_hist = historical.cov()
                inv_cov = np.linalg.pinv(cov_hist)
                diff = current - mean_hist
                turb = np.sqrt(diff @ inv_cov @ diff)
                turbulence.append(turb)
            except:
                turbulence.append(np.nan)
        else:
            turbulence.append(np.nan)
    df['turbulence_index'] = turbulence
    
    # 8. FEATURES TEMPORELLES
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_month'] = df['Date'].dt.day
    df['week_of_year'] = df['Date'].dt.isocalendar().week
    
    # Encodage cyclique
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
    
    # Indicateurs spéciaux
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_quarter_end'] = df['Date'].dt.is_quarter_end.astype(int)
    
    return df

# 3. HMM FEATURES
def add_hmm_features(train_df, test_df, price_cols, n_states=3):
    """HMM avec nombre d'états ajustable"""
    train_returns = train_df[price_cols[:15]].pct_change().fillna(0)
    train_returns = train_returns.replace([np.inf, -np.inf], 0)
    
    test_returns = test_df[price_cols[:15]].pct_change().fillna(0)
    test_returns = test_returns.replace([np.inf, -np.inf], 0)
    
    try:
        hmm_model = hmm.GaussianHMM(
            n_components=n_states, 
            covariance_type="diag",
            n_iter=100,
            random_state=RANDOM_STATE
        )
        hmm_model.fit(train_returns[1:])
        
        train_proba = hmm_model.predict_proba(train_returns[1:])
        test_proba = hmm_model.predict_proba(test_returns)
        
        for i in range(n_states):
            train_df[f'hmm_state_{i}'] = np.concatenate([[0], train_proba[:, i]])
            test_df[f'hmm_state_{i}'] = test_proba[:, i]
            
    except Exception as e:
        print(f"  Erreur HMM: {e}")
        for i in range(n_states):
            train_df[f'hmm_state_{i}'] = 0
            test_df[f'hmm_state_{i}'] = 0
    
    return train_df, test_df

# 4. FONCTION OBJECTIVE OPTUNA
def objective(trial):
    """Fonction objective pour optimisation Bayésienne"""
    
    # Hyperparamètres à optimiser
    params = {
        # XGBoost
        'xgb_n_estimators': trial.suggest_int('xgb_n_estimators', 400, 800),
        'xgb_max_depth': trial.suggest_int('xgb_max_depth', 4, 10),
        'xgb_learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1, log=True),
        'xgb_subsample': trial.suggest_float('xgb_subsample', 0.6, 0.95),
        'xgb_colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 0.95),
        'xgb_gamma': trial.suggest_float('xgb_gamma', 0.0, 0.5),
        'xgb_reg_alpha': trial.suggest_float('xgb_reg_alpha', 0.0, 2.0),
        'xgb_reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.0, 2.0),
        'xgb_min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
        
        # LightGBM
        'lgb_n_estimators': trial.suggest_int('lgb_n_estimators', 400, 800),
        'lgb_num_leaves': trial.suggest_int('lgb_num_leaves', 20, 60),
        'lgb_learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.1, log=True),
        'lgb_feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.5, 0.95),
        'lgb_bagging_fraction': trial.suggest_float('lgb_bagging_fraction', 0.5, 0.95),
        'lgb_bagging_freq': trial.suggest_int('lgb_bagging_freq', 1, 10),
        'lgb_min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 40),
        'lgb_lambda_l1': trial.suggest_float('lgb_lambda_l1', 0.0, 2.0),
        'lgb_lambda_l2': trial.suggest_float('lgb_lambda_l2', 0.0, 2.0),
        
        # CatBoost
        'cat_iterations': trial.suggest_int('cat_iterations', 400, 800),
        'cat_depth': trial.suggest_int('cat_depth', 4, 8),
        'cat_learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.1, log=True),
        'cat_l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1.0, 15.0),
        'cat_border_count': trial.suggest_int('cat_border_count', 32, 255),
        
        # Extra Trees
        'extra_n_estimators': trial.suggest_int('extra_n_estimators', 200, 600),
        'extra_max_depth': trial.suggest_int('extra_max_depth', 10, 25),
        'extra_min_samples_split': trial.suggest_int('extra_min_samples_split', 5, 25),
        'extra_min_samples_leaf': trial.suggest_int('extra_min_samples_leaf', 2, 12),
        'extra_max_features': trial.suggest_categorical('extra_max_features', ['sqrt', 'log2', None]),
        
        # Feature selection
        'n_features': trial.suggest_int('n_features', 250, 350),
        'hmm_n_states': trial.suggest_int('hmm_n_states', 3, 5),
        
        # Post-processing
        'adjustment_strength': trial.suggest_float('adjustment_strength', 0.3, 0.6),
        'persistence_factor': trial.suggest_float('persistence_factor', 1.2, 1.5),
        'turbulence_threshold': trial.suggest_float('turbulence_threshold', 0.8, 0.9),
    }
    
    # Créer features
    train_features = create_advanced_features(train.copy())
    test_features = create_advanced_features(test.copy())
    
    # Ajouter HMM
    price_cols = [col for col in train.columns if col not in ['Date', 'Market_Regime']]
    train_features, test_features = add_hmm_features(
        train_features, test_features, price_cols, params['hmm_n_states']
    )
    
    # Préparer données
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_features['Market_Regime'])
    
    feature_cols = [col for col in train_features.columns if col not in ['Date', 'Market_Regime']]
    X = train_features[feature_cols]
    
    # Nettoyer
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    valid_cols = X.columns[X.isnull().sum() < len(X) * 0.5]
    X = X[valid_cols]
    
    # Supprimer premières lignes
    min_valid_row = 60
    X = X.iloc[min_valid_row:].reset_index(drop=True)
    y = y[min_valid_row:]
    
    # Sélection de features
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Top features
    n_features = min(params['n_features'], len(feature_importances))
    top_features = feature_importances.head(n_features)['feature'].tolist()
    
    # Toujours inclure features clés
    key_features = [col for col in X.columns if any(x in col for x in ['hmm_', 'turbulence', 'avg_corr', 'wavelet'])]
    selected_features = list(set(top_features + key_features))[:params['n_features']]
    
    X_selected = X[selected_features]
    
    # Normalisation
    scaler = RobustScaler()
    X_selected_scaled = scaler.fit_transform(X_selected)
    
    # Créer modèles
    models = {
        'xgb': xgb.XGBClassifier(
            n_estimators=params['xgb_n_estimators'],
            max_depth=params['xgb_max_depth'],
            learning_rate=params['xgb_learning_rate'],
            subsample=params['xgb_subsample'],
            colsample_bytree=params['xgb_colsample_bytree'],
            gamma=params['xgb_gamma'],
            reg_alpha=params['xgb_reg_alpha'],
            reg_lambda=params['xgb_reg_lambda'],
            min_child_weight=params['xgb_min_child_weight'],
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric='mlogloss',
            use_label_encoder=False
        ),
        
        'lgb': lgb.LGBMClassifier(
            n_estimators=params['lgb_n_estimators'],
            num_leaves=params['lgb_num_leaves'],
            learning_rate=params['lgb_learning_rate'],
            feature_fraction=params['lgb_feature_fraction'],
            bagging_fraction=params['lgb_bagging_fraction'],
            bagging_freq=params['lgb_bagging_freq'],
            min_child_samples=params['lgb_min_child_samples'],
            lambda_l1=params['lgb_lambda_l1'],
            lambda_l2=params['lgb_lambda_l2'],
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1
        ),
        
        'cat': CatBoostClassifier(
            iterations=params['cat_iterations'],
            learning_rate=params['cat_learning_rate'],
            depth=params['cat_depth'],
            l2_leaf_reg=params['cat_l2_leaf_reg'],
            border_count=params['cat_border_count'],
            random_state=RANDOM_STATE,
            verbose=False,
            thread_count=-1
        ),
        
        'extra': ExtraTreesClassifier(
            n_estimators=params['extra_n_estimators'],
            max_depth=params['extra_max_depth'],
            min_samples_split=params['extra_min_samples_split'],
            min_samples_leaf=params['extra_min_samples_leaf'],
            max_features=params['extra_max_features'],
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
    
    # Nettoyage mémoire
    gc.collect()
    
    return mean_score

# 5. FONCTION PRINCIPALE
def main():
    print("\n[2] Création de l'étude Optuna...")
    
    # Créer étude
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_STATE, n_startup_trials=20),
        study_name='ultimate_final_optimization'
    )
    
    # Callback pour sauvegarder
    def save_callback(study, trial):
        if trial.number % 10 == 0:
            # Sauvegarder étude
            with open(f'optuna_study_trial_{trial.number}.pkl', 'wb') as f:
                pickle.dump(study, f)
            
            # Sauvegarder meilleurs params
            if study.best_trial:
                with open(f'best_params_trial_{trial.number}.json', 'w') as f:
                    json.dump({
                        'best_score': study.best_value,
                        'best_params': study.best_params,
                        'n_trials': trial.number
                    }, f, indent=2)
                
                print(f"\n[Progress] Trial {trial.number} - Best score so far: {study.best_value:.4f}")
    
    # Optimiser
    print(f"\n[3] Lancement de l'optimisation ({N_TRIALS} trials)...")
    print("Cela peut prendre plusieurs heures...")
    
    start_time = datetime.now()
    
    try:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            callbacks=[save_callback],
            show_progress_bar=True,
            gc_after_trial=True
        )
    except KeyboardInterrupt:
        print("\nOptimisation interrompue par l'utilisateur.")
    
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
    
    # Sauvegarder résultats finaux
    final_results = {
        'best_score': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'duration': str(duration),
        'timestamp': datetime.now().isoformat()
    }
    
    with open('final_best_params.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Sauvegarder étude complète
    with open('optuna_study_final.pkl', 'wb') as f:
        pickle.dump(study, f)
    
    # Visualisations
    print("\n[4] Génération des visualisations...")
    
    try:
        import optuna.visualization as vis
        
        # Historique
        fig = vis.plot_optimization_history(study)
        fig.write_html('optimization_history.html')
        
        # Importance des paramètres
        fig = vis.plot_param_importances(study)
        fig.write_html('param_importances.html')
        
        # Contour plots pour les paires importantes
        fig = vis.plot_contour(study, params=['xgb_learning_rate', 'xgb_n_estimators'])
        fig.write_html('contour_xgb.html')
        
        print("Visualisations sauvegardées:")
        print("  - optimization_history.html")
        print("  - param_importances.html")
        print("  - contour_xgb.html")
    except:
        print("Visualisations non générées (plotly requis)")
    
    print("\n" + "="*80)
    print("PROCESSUS COMPLET TERMINÉ!")
    print("\nProchaines étapes:")
    print("1. Examiner les résultats dans 'final_best_params.json'")
    print("2. Utiliser create_final_model.py pour entraîner avec les meilleurs paramètres")
    print("3. Soumettre les prédictions sur Kaggle")
    print("="*80)

if __name__ == "__main__":
    main()