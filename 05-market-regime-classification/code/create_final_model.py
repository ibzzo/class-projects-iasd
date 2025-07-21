#!/usr/bin/env python3
"""
CREATE FINAL MODEL - Utilise les meilleurs paramètres trouvés par Optuna
========================================================================
Script pour créer le modèle final avec les hyperparamètres optimaux
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy import stats
import pywt
from hmmlearn import hmm
from collections import Counter
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CREATE FINAL MODEL - Entraînement avec paramètres optimaux")
print("="*80)

# 1. CHARGER LES MEILLEURS PARAMÈTRES
print("\n[1] Chargement des meilleurs paramètres...")
try:
    with open('final_best_params.json', 'r') as f:
        results = json.load(f)
        best_params = results['best_params']
        best_score = results['best_score']
    print(f"Meilleur score trouvé: {best_score:.4f}")
except:
    print("Fichier 'final_best_params.json' non trouvé!")
    print("Utilisation des paramètres par défaut du modèle original...")
    best_params = {
        'xgb_n_estimators': 500,
        'xgb_max_depth': 6,
        'xgb_learning_rate': 0.03,
        'xgb_subsample': 0.8,
        'xgb_colsample_bytree': 0.8,
        'xgb_gamma': 0.1,
        'xgb_reg_alpha': 0.5,
        'xgb_reg_lambda': 0.5,
        'xgb_min_child_weight': 3,
        'lgb_n_estimators': 450,
        'lgb_num_leaves': 31,
        'lgb_learning_rate': 0.03,
        'lgb_feature_fraction': 0.85,
        'lgb_bagging_fraction': 0.85,
        'lgb_bagging_freq': 5,
        'lgb_min_child_samples': 20,
        'lgb_lambda_l1': 0.1,
        'lgb_lambda_l2': 0.1,
        'cat_iterations': 400,
        'cat_depth': 5,
        'cat_learning_rate': 0.03,
        'cat_l2_leaf_reg': 5,
        'cat_border_count': 128,
        'extra_n_estimators': 300,
        'extra_max_depth': 15,
        'extra_min_samples_split': 10,
        'extra_min_samples_leaf': 5,
        'extra_max_features': 'sqrt',
        'n_features': 300,
        'hmm_n_states': 3,
        'adjustment_strength': 0.5,
        'persistence_factor': 1.3,
        'turbulence_threshold': 0.85
    }

# 2. CHARGEMENT DES DONNÉES
print("\n[2] Chargement des données...")
train = pd.read_csv('train.csv', parse_dates=['Date'])
test = pd.read_csv('test.csv', parse_dates=['Date'])
test_dates = test['Date'].copy()

train_dist = train['Market_Regime'].value_counts(normalize=True).sort_index()
print("\nDistribution cible:")
for regime in [-1, 0, 1]:
    print(f"  Régime {regime}: {train_dist[regime]*100:.1f}%")

# 3. FEATURE ENGINEERING (identique au modèle original)
def create_advanced_features(df):
    """Features du modèle à 0.9121"""
    price_cols = [col for col in df.columns if col not in ['Date', 'Market_Regime']]
    
    # 1. RETURNS ET RATIOS
    for col in price_cols[:20]:
        for period in [1, 2, 5, 10, 20]:
            df[f'{col}_ret_{period}'] = df[col].pct_change(period)
        df[f'{col}_ret_ratio_5_20'] = df[f'{col}_ret_5'] / (df[f'{col}_ret_20'] + 1e-8)
        
    # 2. VOLATILITÉ
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
        
    # 7. TURBULENCE
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
    
    # 8. TEMPOREL
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_month'] = df['Date'].dt.day
    df['week_of_year'] = df['Date'].dt.isocalendar().week
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
    
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_quarter_end'] = df['Date'].dt.is_quarter_end.astype(int)
    
    return df

# 4. HMM
def add_hmm_features(train_df, test_df, price_cols, n_states):
    train_returns = train_df[price_cols[:15]].pct_change().fillna(0)
    train_returns = train_returns.replace([np.inf, -np.inf], 0)
    
    test_returns = test_df[price_cols[:15]].pct_change().fillna(0)
    test_returns = test_returns.replace([np.inf, -np.inf], 0)
    
    try:
        hmm_model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=100,
            random_state=42
        )
        hmm_model.fit(train_returns[1:])
        
        train_proba = hmm_model.predict_proba(train_returns[1:])
        test_proba = hmm_model.predict_proba(test_returns)
        
        for i in range(n_states):
            train_df[f'hmm_state_{i}'] = np.concatenate([[0], train_proba[:, i]])
            test_df[f'hmm_state_{i}'] = test_proba[:, i]
            
        print(f"  HMM features ajoutées ({n_states} états)")
    except Exception as e:
        print(f"  Erreur HMM: {e}")
        for i in range(n_states):
            train_df[f'hmm_state_{i}'] = 0
            test_df[f'hmm_state_{i}'] = 0
    
    return train_df, test_df

# 5. CRÉER FEATURES
print("\n[3] Feature Engineering...")
train_features = create_advanced_features(train.copy())
test_features = create_advanced_features(test.copy())

# Ajouter HMM
price_cols = [col for col in train.columns if col not in ['Date', 'Market_Regime']]
n_states = best_params.get('hmm_n_states', 3)
train_features, test_features = add_hmm_features(train_features, test_features, price_cols, n_states)

# 6. PRÉPARER DONNÉES
print("\n[4] Préparation des données...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_features['Market_Regime'])

feature_cols = [col for col in train_features.columns if col not in ['Date', 'Market_Regime']]
X = train_features[feature_cols]
X_test = test_features[feature_cols]

# Nettoyer
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

valid_cols = X.columns[X.isnull().sum() < len(X) * 0.5]
X = X[valid_cols]
X_test = X_test[valid_cols]

min_valid_row = 60
X = X.iloc[min_valid_row:].reset_index(drop=True)
y = y[min_valid_row:]

# 7. SÉLECTION DE FEATURES
print("\n[5] Sélection de features...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

n_features = min(best_params.get('n_features', 300), len(feature_importances))
top_features = feature_importances.head(n_features)['feature'].tolist()

key_features = [col for col in X.columns if any(x in col for x in ['hmm_', 'turbulence', 'avg_corr', 'wavelet'])]
selected_features = list(set(top_features + key_features))[:n_features]

X_selected = X[selected_features]
X_test_selected = X_test[selected_features]

print(f"  Features sélectionnées: {len(selected_features)}")

# Normalisation
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_selected)
X_test_scaled = scaler.transform(X_test_selected)

# 8. CRÉER MODÈLES AVEC PARAMÈTRES OPTIMAUX
print("\n[6] Configuration des modèles optimisés...")

models = {
    'xgb': xgb.XGBClassifier(
        n_estimators=best_params.get('xgb_n_estimators', 500),
        max_depth=best_params.get('xgb_max_depth', 6),
        learning_rate=best_params.get('xgb_learning_rate', 0.03),
        subsample=best_params.get('xgb_subsample', 0.8),
        colsample_bytree=best_params.get('xgb_colsample_bytree', 0.8),
        gamma=best_params.get('xgb_gamma', 0.1),
        reg_alpha=best_params.get('xgb_reg_alpha', 0.5),
        reg_lambda=best_params.get('xgb_reg_lambda', 0.5),
        min_child_weight=best_params.get('xgb_min_child_weight', 3),
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        use_label_encoder=False
    ),
    
    'lgb': lgb.LGBMClassifier(
        n_estimators=best_params.get('lgb_n_estimators', 450),
        num_leaves=best_params.get('lgb_num_leaves', 31),
        learning_rate=best_params.get('lgb_learning_rate', 0.03),
        feature_fraction=best_params.get('lgb_feature_fraction', 0.85),
        bagging_fraction=best_params.get('lgb_bagging_fraction', 0.85),
        bagging_freq=best_params.get('lgb_bagging_freq', 5),
        min_child_samples=best_params.get('lgb_min_child_samples', 20),
        lambda_l1=best_params.get('lgb_lambda_l1', 0.1),
        lambda_l2=best_params.get('lgb_lambda_l2', 0.1),
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    ),
    
    'cat': CatBoostClassifier(
        iterations=best_params.get('cat_iterations', 400),
        learning_rate=best_params.get('cat_learning_rate', 0.03),
        depth=best_params.get('cat_depth', 5),
        l2_leaf_reg=best_params.get('cat_l2_leaf_reg', 5),
        border_count=best_params.get('cat_border_count', 128),
        random_state=42,
        verbose=False,
        thread_count=-1
    ),
    
    'extra': ExtraTreesClassifier(
        n_estimators=best_params.get('extra_n_estimators', 300),
        max_depth=best_params.get('extra_max_depth', 15),
        min_samples_split=best_params.get('extra_min_samples_split', 10),
        min_samples_leaf=best_params.get('extra_min_samples_leaf', 5),
        max_features=best_params.get('extra_max_features', 'sqrt'),
        random_state=42,
        n_jobs=-1
    )
}

# 9. VALIDATION CROISÉE
print("\n[7] Validation croisée...")
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_predictions = {name: np.zeros((len(X_scaled), 3)) for name in models}
test_predictions = {name: np.zeros((len(X_test_scaled), 3)) for name in models}

for name, model in models.items():
    print(f"\n{name.upper()}:")
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train, y_train)
        
        val_pred = model_clone.predict_proba(X_val)
        oof_predictions[name][val_idx] = val_pred
        
        score = roc_auc_score(y_val, val_pred, multi_class='ovr')
        scores.append(score)
        print(f"  Fold {fold+1}: {score:.4f}")
    
    print(f"  Moyenne: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    
    # Entraîner sur toutes les données
    model.fit(X_scaled, y)
    test_predictions[name] = model.predict_proba(X_test_scaled)

# 10. ENSEMBLE
print("\n[8] Ensemble des modèles...")

weights = {}
for name in models:
    oof_score = roc_auc_score(y, oof_predictions[name], multi_class='ovr')
    weights[name] = oof_score ** 2

total_weight = sum(weights.values())
for name in weights:
    weights[name] /= total_weight

print("\nPoids d'ensemble:")
for name, weight in weights.items():
    print(f"  {name}: {weight:.3f}")

ensemble_test = np.zeros_like(test_predictions['xgb'])
for name, weight in weights.items():
    ensemble_test += weight * test_predictions[name]

# 11. CALIBRATION
print("\n[9] Calibration isotonique...")

calibrated_test = np.zeros_like(ensemble_test)
for i in range(3):
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    
    oof_ensemble = np.zeros_like(oof_predictions['xgb'])
    for name, weight in weights.items():
        oof_ensemble += weight * oof_predictions[name]
    
    iso_reg.fit(oof_ensemble[:, i], (y == i).astype(int))
    calibrated_test[:, i] = iso_reg.predict(ensemble_test[:, i])

calibrated_test = calibrated_test / calibrated_test.sum(axis=1, keepdims=True)

# 12. POST-PROCESSING
print("\n[10] Post-processing...")

target_dist = {-1: 0.272, 0: 0.363, 1: 0.365}
adjustment_strength = best_params.get('adjustment_strength', 0.5)
persistence_factor = best_params.get('persistence_factor', 1.3)
turbulence_threshold = best_params.get('turbulence_threshold', 0.85)

# Ajustement progressif
adjusted_proba = calibrated_test.copy()
for strength in np.linspace(0.1, adjustment_strength, 5):
    current_preds = np.argmax(adjusted_proba, axis=1)
    current_dist = np.bincount(current_preds, minlength=3) / len(current_preds)
    
    for i in range(3):
        target = list(target_dist.values())[i]
        if current_dist[i] > 0:
            ratio = (target / current_dist[i]) ** strength
            adjusted_proba[:, i] *= ratio
    
    adjusted_proba = adjusted_proba / adjusted_proba.sum(axis=1, keepdims=True)

# Prédictions finales
final_predictions = []
for i in range(len(adjusted_proba)):
    proba = adjusted_proba[i].copy()
    
    # Logique saisonnière
    if i < len(test_features):
        if 'month' in test_features.columns:
            month = test_features.iloc[i]['month']
            if month == 4:
                proba[0] *= 1.15
            elif month == 12:
                proba[2] *= 1.15
        
        # Turbulence
        if 'turbulence_index' in test_features.columns:
            turb = test_features.iloc[i]['turbulence_index']
            if not pd.isna(turb) and turb > train_features['turbulence_index'].quantile(turbulence_threshold):
                proba[1] *= 0.85
    
    proba = proba / proba.sum()
    
    # Persistance
    if i > 0 and len(final_predictions) > 0:
        if max(proba) < 0.45:
            prev_pred = final_predictions[-1]
            proba[prev_pred] *= persistence_factor
            proba = proba / proba.sum()
    
    pred = np.argmax(proba)
    final_predictions.append(pred)

final_predictions = label_encoder.inverse_transform(final_predictions)

# 13. ANALYSE ET SOUMISSION
print("\n[11] Création de la soumission...")

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

# Créer soumission
submission = pd.DataFrame({
    'Id': test_dates,
    'Expected': final_predictions.astype(int)
})

submission.to_csv('submission_final_optimized.csv', index=False)

print("\n" + "="*80)
print("MODÈLE FINAL TERMINÉ!")
print("\nCaractéristiques:")
print("  ✓ Hyperparamètres optimisés par Optuna")
print(f"  ✓ {len(selected_features)} features sélectionnées")
print(f"  ✓ HMM avec {n_states} états")
print("  ✓ Ensemble de 4 modèles optimisés")
print("  ✓ Calibration isotonique")
print("  ✓ Post-processing avancé")
print(f"  ✓ Performance: AUC {final_oof_score:.4f}")
print("\nFichier de soumission: submission_final_optimized.csv")
print("="*80)