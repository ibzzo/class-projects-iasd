#!/usr/bin/env python3
"""
ULTIMATE OPTUNA OPTIMIZED MODEL - Optimisation Bayésienne Complète
Objectif: AUC > 0.92 avec recherche d'hyperparamètres exhaustive
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy import stats
from scipy.signal import hilbert
import pywt
from hmmlearn import hmm
from collections import Counter
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("="*80)
print("ULTIMATE OPTUNA OPTIMIZED MODEL - Recherche Bayésienne")
print("="*80)

# 1. CHARGEMENT DES DONNÉES
print("\n[1] Chargement des données...")
train = pd.read_csv('train.csv', parse_dates=['Date'])
test = pd.read_csv('test.csv', parse_dates=['Date'])
test_dates = test['Date'].copy()

train_dist = train['Market_Regime'].value_counts(normalize=True).sort_index()
print("\nDistribution cible:")
for regime in [-1, 0, 1]:
    print(f"  Régime {regime}: {train_dist[regime]*100:.1f}%")

# 2. FEATURE ENGINEERING AVANCÉ (même que l'original mais avec paramètres)
def create_advanced_features(df, params=None):
    """Features avancées avec paramètres ajustables"""
    if params is None:
        params = {
            'n_price_cols': 20,
            'n_return_periods': [1, 2, 5, 10, 20],
            'n_vol_windows': [5, 10, 20, 30],
            'n_wavelet_windows': [32, 64],
            'n_corr_cols': 10,
            'n_rsi_period': 14
        }
    
    price_cols = [col for col in df.columns if col not in ['Date', 'Market_Regime']]
    
    # 1. RETURNS ET RATIOS
    print("  - Returns et ratios...")
    for col in price_cols[:params['n_price_cols']]:
        # Returns classiques
        for period in params['n_return_periods']:
            df[f'{col}_ret_{period}'] = df[col].pct_change(period)
            
        # Ratios de returns
        df[f'{col}_ret_ratio_5_20'] = df[f'{col}_ret_5'] / (df[f'{col}_ret_20'] + 1e-8)
        
    # 2. VOLATILITÉ AVANCÉE
    print("  - Volatilité avancée...")
    for col in price_cols[:15]:
        returns = df[col].pct_change()
        
        # Volatilité classique
        for window in params['n_vol_windows']:
            df[f'{col}_vol_{window}'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Volatilité EWMA
        df[f'{col}_ewma_vol'] = returns.ewm(span=20).std() * np.sqrt(252)
        
        # Ratio de volatilité
        df[f'{col}_vol_ratio'] = df[f'{col}_vol_5'] / (df[f'{col}_vol_20'] + 1e-8)
        
    # 3. WAVELETS SIMPLIFIÉS
    print("  - Wavelets...")
    for col in price_cols[:10]:
        try:
            # Décomposition simple
            for window in params['n_wavelet_windows']:
                df[f'{col}_wavelet_{window}'] = df[col].rolling(window).apply(
                    lambda x: np.std(pywt.dwt(x, 'db4')[0]) if len(x) == window else np.nan
                )
        except:
            pass
    
    # 4. MICROSTRUCTURE
    print("  - Microstructure...")
    for col in price_cols[:10]:
        returns = df[col].pct_change()
        
        # Realized volatility
        df[f'{col}_realized_vol'] = returns.rolling(20).apply(
            lambda x: np.sqrt(np.sum(x**2))
        )
        
        # Nombre de changements de signe
        df[f'{col}_sign_changes'] = returns.rolling(20).apply(
            lambda x: np.sum(np.diff(np.sign(x)) != 0)
        )
        
    # 5. CORRÉLATIONS AVANCÉES
    print("  - Corrélations...")
    for window in [20, 30]:
        corr_features = []
        for i in range(len(df)):
            if i >= window:
                corr_matrix = df[price_cols[:params['n_corr_cols']]].iloc[i-window:i].corr()
                # Moyenne des corrélations
                upper_tri = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                avg_corr = np.nanmean(upper_tri)
                # Max eigenvalue (mesure de concentration)
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
    print("  - Indicateurs techniques...")
    for col in price_cols[:10]:
        # RSI
        delta = df[col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(params['n_rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(params['n_rsi_period']).mean()
        rs = gain / (loss + 1e-8)
        df[f'{col}_rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma = df[col].rolling(20).mean()
        std = df[col].rolling(20).std()
        df[f'{col}_bb_upper'] = (df[col] - (sma + 2*std)) / df[col]
        df[f'{col}_bb_lower'] = ((sma - 2*std) - df[col]) / df[col]
        
    # 7. TURBULENCE INDEX
    print("  - Turbulence index...")
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
    print("  - Features temporelles...")
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

# 3. HIDDEN MARKOV MODEL ROBUSTE
def add_hmm_features(train_df, test_df, price_cols, n_states=3):
    """Ajoute les features HMM avec nombre d'états variable"""
    # Préparer les données
    train_returns = train_df[price_cols[:15]].pct_change().fillna(0)
    train_returns = train_returns.replace([np.inf, -np.inf], 0)
    
    test_returns = test_df[price_cols[:15]].pct_change().fillna(0)
    test_returns = test_returns.replace([np.inf, -np.inf], 0)
    
    # HMM avec n états
    try:
        hmm_model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", 
                                    n_iter=100, random_state=42)
        hmm_model.fit(train_returns[1:])
        
        # Prédictions
        train_proba = hmm_model.predict_proba(train_returns[1:])
        test_proba = hmm_model.predict_proba(test_returns)
        
        # Ajouter aux dataframes
        for i in range(n_states):
            train_df[f'hmm_state_{i}'] = np.concatenate([[0], train_proba[:, i]])
            test_df[f'hmm_state_{i}'] = test_proba[:, i]
            
        print(f"  HMM features ajoutées avec succès ({n_states} états)")
    except Exception as e:
        print(f"  Erreur HMM: {e}")
        for i in range(n_states):
            train_df[f'hmm_state_{i}'] = 0
            test_df[f'hmm_state_{i}'] = 0
    
    return train_df, test_df

# 4. FONCTION OBJECTIVE OPTUNA
def objective(trial):
    """Fonction objective pour Optuna"""
    
    # Hyperparamètres à optimiser
    params = {
        # XGBoost
        'xgb_n_estimators': trial.suggest_int('xgb_n_estimators', 300, 800),
        'xgb_max_depth': trial.suggest_int('xgb_max_depth', 4, 10),
        'xgb_learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1, log=True),
        'xgb_subsample': trial.suggest_float('xgb_subsample', 0.6, 0.95),
        'xgb_colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 0.95),
        'xgb_gamma': trial.suggest_float('xgb_gamma', 0.0, 0.5),
        'xgb_reg_alpha': trial.suggest_float('xgb_reg_alpha', 0.0, 2.0),
        'xgb_reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.0, 2.0),
        
        # LightGBM
        'lgb_n_estimators': trial.suggest_int('lgb_n_estimators', 300, 800),
        'lgb_num_leaves': trial.suggest_int('lgb_num_leaves', 20, 50),
        'lgb_learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.1, log=True),
        'lgb_feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.6, 0.95),
        'lgb_bagging_fraction': trial.suggest_float('lgb_bagging_fraction', 0.6, 0.95),
        'lgb_bagging_freq': trial.suggest_int('lgb_bagging_freq', 1, 10),
        'lgb_min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 30),
        
        # CatBoost
        'cat_iterations': trial.suggest_int('cat_iterations', 300, 800),
        'cat_depth': trial.suggest_int('cat_depth', 4, 8),
        'cat_learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.1, log=True),
        'cat_l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1.0, 10.0),
        
        # Extra Trees
        'extra_n_estimators': trial.suggest_int('extra_n_estimators', 200, 500),
        'extra_max_depth': trial.suggest_int('extra_max_depth', 10, 25),
        'extra_min_samples_split': trial.suggest_int('extra_min_samples_split', 5, 20),
        'extra_min_samples_leaf': trial.suggest_int('extra_min_samples_leaf', 2, 10),
        
        # Feature engineering
        'n_features': trial.suggest_int('n_features', 200, 400),
        'hmm_n_states': trial.suggest_int('hmm_n_states', 3, 5)
    }
    
    # Créer les modèles avec ces paramètres
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
            random_state=42,
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
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        ),
        
        'cat': CatBoostClassifier(
            iterations=params['cat_iterations'],
            learning_rate=params['cat_learning_rate'],
            depth=params['cat_depth'],
            l2_leaf_reg=params['cat_l2_leaf_reg'],
            random_state=42,
            verbose=False
        ),
        
        'extra': ExtraTreesClassifier(
            n_estimators=params['extra_n_estimators'],
            max_depth=params['extra_max_depth'],
            min_samples_split=params['extra_min_samples_split'],
            min_samples_leaf=params['extra_min_samples_leaf'],
            random_state=42,
            n_jobs=-1
        )
    }
    
    # Feature engineering avec HMM states variable
    global train_features_base, test_features_base, price_cols
    train_features = train_features_base.copy()
    test_features = test_features_base.copy()
    train_features, test_features = add_hmm_features(train_features, test_features, price_cols, params['hmm_n_states'])
    
    # Préparer les données
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_features['Market_Regime'])
    
    feature_cols = [col for col in train_features.columns 
                    if col not in ['Date', 'Market_Regime']]
    X = train_features[feature_cols]
    
    # Nettoyer
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Supprimer les colonnes avec trop de NaN
    valid_cols = X.columns[X.isnull().sum() < len(X) * 0.5]
    X = X[valid_cols]
    
    # Supprimer les premières lignes
    min_valid_row = 60
    X = X.iloc[min_valid_row:].reset_index(drop=True)
    y = y[min_valid_row:]
    
    # Sélection de features par Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Sélectionner les meilleures features
    n_features = min(params['n_features'], len(feature_importances))
    top_features = feature_importances.head(n_features)['feature'].tolist()
    
    # Toujours inclure certaines features clés
    key_features = [col for col in X.columns if any(x in col for x in ['hmm_', 'turbulence', 'avg_corr', 'wavelet'])]
    selected_features = list(set(top_features + key_features))[:params['n_features']]
    
    X_selected = X[selected_features]
    
    # Validation croisée
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    scores = []
    
    for name, model in models.items():
        model_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected, y)):
            X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Entraîner
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_train, y_train)
            
            # Prédire
            val_pred = model_clone.predict_proba(X_val)
            
            # Score
            score = roc_auc_score(y_val, val_pred, multi_class='ovr')
            model_scores.append(score)
        
        scores.extend(model_scores)
    
    # Retourner la moyenne des scores
    return np.mean(scores)

# 5. PRÉPARATION DES DONNÉES DE BASE
print("\n[2] Feature Engineering de base...")
train_features_base = create_advanced_features(train.copy())
test_features_base = create_advanced_features(test.copy())
price_cols = [col for col in train.columns if col not in ['Date', 'Market_Regime']]

# 6. OPTIMISATION OPTUNA
print("\n[3] Lancement de l'optimisation Optuna...")
print("  Cela peut prendre plusieurs heures...")

# Créer l'étude Optuna
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    study_name='ultimate_robust_optimization'
)

# Optimiser
n_trials = 100  # Augmenter pour une recherche plus exhaustive
study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

print(f"\n[4] Optimisation terminée!")
print(f"  Meilleur score: {study.best_value:.4f}")
print(f"  Meilleurs paramètres:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")

# 7. ENTRAÎNEMENT FINAL AVEC LES MEILLEURS PARAMÈTRES
print("\n[5] Entraînement final avec les meilleurs paramètres...")

best_params = study.best_params

# Recréer les features avec les meilleurs paramètres
train_features = train_features_base.copy()
test_features = test_features_base.copy()
train_features, test_features = add_hmm_features(train_features, test_features, price_cols, best_params['hmm_n_states'])

# Préparer les données
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_features['Market_Regime'])

feature_cols = [col for col in train_features.columns 
                if col not in ['Date', 'Market_Regime']]
X = train_features[feature_cols]
X_test = test_features[feature_cols]

# Nettoyer
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

# Sélection de features
valid_cols = X.columns[X.isnull().sum() < len(X) * 0.5]
X = X[valid_cols]
X_test = X_test[valid_cols]

min_valid_row = 60
X = X.iloc[min_valid_row:].reset_index(drop=True)
y = y[min_valid_row:]

# Sélection finale
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

n_features = min(best_params['n_features'], len(feature_importances))
top_features = feature_importances.head(n_features)['feature'].tolist()
key_features = [col for col in X.columns if any(x in col for x in ['hmm_', 'turbulence', 'avg_corr', 'wavelet'])]
selected_features = list(set(top_features + key_features))[:best_params['n_features']]

X_selected = X[selected_features]
X_test_selected = X_test[selected_features]

print(f"  Features sélectionnées: {len(selected_features)}")

# Créer les modèles optimisés
final_models = {
    'xgb': xgb.XGBClassifier(
        n_estimators=best_params['xgb_n_estimators'],
        max_depth=best_params['xgb_max_depth'],
        learning_rate=best_params['xgb_learning_rate'],
        subsample=best_params['xgb_subsample'],
        colsample_bytree=best_params['xgb_colsample_bytree'],
        gamma=best_params['xgb_gamma'],
        reg_alpha=best_params['xgb_reg_alpha'],
        reg_lambda=best_params['xgb_reg_lambda'],
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        use_label_encoder=False
    ),
    
    'lgb': lgb.LGBMClassifier(
        n_estimators=best_params['lgb_n_estimators'],
        num_leaves=best_params['lgb_num_leaves'],
        learning_rate=best_params['lgb_learning_rate'],
        feature_fraction=best_params['lgb_feature_fraction'],
        bagging_fraction=best_params['lgb_bagging_fraction'],
        bagging_freq=best_params['lgb_bagging_freq'],
        min_child_samples=best_params['lgb_min_child_samples'],
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    ),
    
    'cat': CatBoostClassifier(
        iterations=best_params['cat_iterations'],
        learning_rate=best_params['cat_learning_rate'],
        depth=best_params['cat_depth'],
        l2_leaf_reg=best_params['cat_l2_leaf_reg'],
        random_state=42,
        verbose=False
    ),
    
    'extra': ExtraTreesClassifier(
        n_estimators=best_params['extra_n_estimators'],
        max_depth=best_params['extra_max_depth'],
        min_samples_split=best_params['extra_min_samples_split'],
        min_samples_leaf=best_params['extra_min_samples_leaf'],
        random_state=42,
        n_jobs=-1
    )
}

# 8. VALIDATION CROISÉE FINALE
print("\n[6] Validation croisée finale...")

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_predictions = {name: np.zeros((len(X_selected), 3)) for name in final_models}
test_predictions = {name: np.zeros((len(X_test_selected), 3)) for name in final_models}

for name, model in final_models.items():
    print(f"\n{name.upper()}:")
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected, y)):
        X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Entraîner
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train, y_train)
        
        # Prédire
        val_pred = model_clone.predict_proba(X_val)
        oof_predictions[name][val_idx] = val_pred
        
        # Score
        score = roc_auc_score(y_val, val_pred, multi_class='ovr')
        scores.append(score)
        print(f"  Fold {fold+1}: {score:.4f}")
    
    print(f"  Moyenne: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    
    # Entraîner sur toutes les données
    model.fit(X_selected, y)
    test_predictions[name] = model.predict_proba(X_test_selected)

# 9. ENSEMBLE AVEC POIDS OPTIMAUX
print("\n[7] Ensemble des modèles...")

# Poids basés sur les performances OOF
weights = {}
for name in final_models:
    oof_score = roc_auc_score(y, oof_predictions[name], multi_class='ovr')
    weights[name] = oof_score ** 2  # Poids quadratique

# Normaliser
total_weight = sum(weights.values())
for name in weights:
    weights[name] /= total_weight

print("\nPoids d'ensemble:")
for name, weight in weights.items():
    print(f"  {name}: {weight:.3f}")

# Ensemble
ensemble_test = np.zeros_like(test_predictions['xgb'])
for name, weight in weights.items():
    ensemble_test += weight * test_predictions[name]

# 10. CALIBRATION ET POST-PROCESSING
print("\n[8] Calibration et post-processing...")

from sklearn.isotonic import IsotonicRegression

# Calibration isotonique
calibrated_test = np.zeros_like(ensemble_test)
for i in range(3):
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    # Utiliser les prédictions OOF moyennes pour la calibration
    oof_ensemble = np.zeros_like(oof_predictions['xgb'])
    for name, weight in weights.items():
        oof_ensemble += weight * oof_predictions[name]
    
    iso_reg.fit(oof_ensemble[:, i], (y == i).astype(int))
    calibrated_test[:, i] = iso_reg.predict(ensemble_test[:, i])

# Normaliser
calibrated_test = calibrated_test / calibrated_test.sum(axis=1, keepdims=True)

# 11. AJUSTEMENT FINAL OPTIMISÉ
print("\n[9] Ajustement final optimisé...")

# Distribution cible
target_dist = {-1: 0.272, 0: 0.363, 1: 0.365}

# Paramètres d'ajustement optimisables
adjustment_strength = trial.suggest_float('adjustment_strength', 0.1, 0.6) if 'trial' in locals() else 0.4
persistence_factor = trial.suggest_float('persistence_factor', 1.1, 1.5) if 'trial' in locals() else 1.3
turbulence_threshold = trial.suggest_float('turbulence_threshold', 0.8, 0.95) if 'trial' in locals() else 0.85

# Ajustement progressif
adjusted_proba = calibrated_test.copy()
for strength in np.linspace(0.1, adjustment_strength, 5):
    # Calculer la distribution actuelle
    current_preds = np.argmax(adjusted_proba, axis=1)
    current_dist = np.bincount(current_preds, minlength=3) / len(current_preds)
    
    # Ajuster
    for i in range(3):
        target = list(target_dist.values())[i]
        if current_dist[i] > 0:
            ratio = (target / current_dist[i]) ** strength
            adjusted_proba[:, i] *= ratio
    
    # Renormaliser
    adjusted_proba = adjusted_proba / adjusted_proba.sum(axis=1, keepdims=True)

# Prédictions finales avec logique de marché optimisée
final_predictions = []
for i in range(len(adjusted_proba)):
    proba = adjusted_proba[i].copy()
    
    # Logique saisonnière
    if i < len(test_features):
        if 'month' in test_features.columns:
            month = test_features.iloc[i]['month']
            if month == 4:  # Avril
                proba[0] *= 1.15
            elif month == 12:  # Décembre
                proba[2] *= 1.15
        
        # Turbulence
        if 'turbulence_index' in test_features.columns:
            turb = test_features.iloc[i]['turbulence_index']
            if not pd.isna(turb) and turb > train_features['turbulence_index'].quantile(turbulence_threshold):
                proba[1] *= 0.85
    
    # Renormaliser
    proba = proba / proba.sum()
    
    # Persistance optimisée
    if i > 0 and len(final_predictions) > 0:
        if max(proba) < 0.45:
            prev_pred = final_predictions[-1]
            proba[prev_pred] *= persistence_factor
            proba = proba / proba.sum()
    
    pred = np.argmax(proba)
    final_predictions.append(pred)

# Décoder
final_predictions = label_encoder.inverse_transform(final_predictions)

# 12. ANALYSE ET SOUMISSION
print("\n[10] Création de la soumission...")

# Distribution finale
dist = Counter(final_predictions)
print("\nDistribution finale:")
for regime in sorted(dist.keys()):
    pct = dist[regime]/len(final_predictions)*100
    target_pct = target_dist[regime]*100
    print(f"  Régime {regime}: {dist[regime]} ({pct:.1f}%) - Cible: {target_pct:.1f}%")

# Score OOF final
oof_ensemble_final = np.zeros_like(oof_predictions['xgb'])
for name, weight in weights.items():
    oof_ensemble_final += weight * oof_predictions[name]
final_oof_score = roc_auc_score(y, oof_ensemble_final, multi_class='ovr')
print(f"\nScore AUC OOF final: {final_oof_score:.4f}")

# Créer la soumission
submission = pd.DataFrame({
    'Id': test_dates,
    'Expected': final_predictions.astype(int)
})

submission.to_csv('submission_optuna_optimized.csv', index=False)

print("\n" + "="*80)
print("ULTIMATE OPTUNA OPTIMIZED MODEL TERMINÉ!")
print("\nCaractéristiques:")
print("  ✓ Optimisation Bayésienne complète avec Optuna")
print(f"  ✓ {n_trials} essais d'optimisation")
print(f"  ✓ {best_params['n_features']} features sélectionnées")
print(f"  ✓ HMM avec {best_params['hmm_n_states']} états")
print("  ✓ Ensemble de 4 modèles avec hyperparamètres optimaux")
print("  ✓ Calibration isotonique")
print("  ✓ Post-processing optimisé")
print(f"  ✓ Performance: AUC {final_oof_score:.4f}")
print("="*80)

# Sauvegarder les meilleurs paramètres
import json
with open('best_params_optuna.json', 'w') as f:
    json.dump(study.best_params, f, indent=2)
print("\nMeilleurs paramètres sauvegardés dans 'best_params_optuna.json'")