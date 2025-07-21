"""
MODÈLE ULTIMATE ROBUSTE - VERSION STABLE ET PERFORMANTE
=======================================================
Intégration des meilleures innovations avec stabilité garantie
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
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MODÈLE ULTIMATE ROBUSTE - APPROCHE STABLE")
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

# 2. FEATURE ENGINEERING AVANCÉ ET STABLE
print("\n[2] Feature Engineering avancé...")

def create_advanced_features(df):
    """Features avancées mais stables"""
    price_cols = [col for col in df.columns if col not in ['Date', 'Market_Regime']]
    
    # 1. RETURNS ET RATIOS
    print("  - Returns et ratios...")
    for col in price_cols[:20]:
        # Returns classiques
        for period in [1, 2, 5, 10, 20]:
            df[f'{col}_ret_{period}'] = df[col].pct_change(period)
            
        # Ratios de returns
        df[f'{col}_ret_ratio_5_20'] = df[f'{col}_ret_5'] / (df[f'{col}_ret_20'] + 1e-8)
        
    # 2. VOLATILITÉ AVANCÉE
    print("  - Volatilité avancée...")
    for col in price_cols[:15]:
        returns = df[col].pct_change()
        
        # Volatilité classique
        for window in [5, 10, 20, 30]:
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
            for window in [32, 64]:  # Puissances de 2 pour wavelets
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
                corr_matrix = df[price_cols[:10]].iloc[i-window:i].corr()
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
print("\n[3] Hidden Markov Models...")

def add_hmm_features(train_df, test_df, price_cols):
    """Ajoute les features HMM de manière robuste"""
    # Préparer les données
    train_returns = train_df[price_cols[:15]].pct_change().fillna(0)
    train_returns = train_returns.replace([np.inf, -np.inf], 0)
    
    test_returns = test_df[price_cols[:15]].pct_change().fillna(0)
    test_returns = test_returns.replace([np.inf, -np.inf], 0)
    
    # HMM avec 3 états
    try:
        hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="diag", 
                                    n_iter=100, random_state=42)
        hmm_model.fit(train_returns[1:])
        
        # Prédictions
        train_proba = hmm_model.predict_proba(train_returns[1:])
        test_proba = hmm_model.predict_proba(test_returns)
        
        # Ajouter aux dataframes
        for i in range(3):
            train_df[f'hmm_state_{i}'] = np.concatenate([[0], train_proba[:, i]])
            test_df[f'hmm_state_{i}'] = test_proba[:, i]
            
        print("  HMM features ajoutées avec succès")
    except Exception as e:
        print(f"  Erreur HMM: {e}")
        for i in range(3):
            train_df[f'hmm_state_{i}'] = 0
            test_df[f'hmm_state_{i}'] = 0
    
    return train_df, test_df

# Application du feature engineering
train_features = create_advanced_features(train.copy())
test_features = create_advanced_features(test.copy())

# Ajouter HMM
price_cols = [col for col in train.columns if col not in ['Date', 'Market_Regime']]
train_features, test_features = add_hmm_features(train_features, test_features, price_cols)

# 4. SÉLECTION DE FEATURES INTELLIGENTE
print("\n[4] Sélection de features...")

# Préparer les données
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_features['Market_Regime'])

# Sélectionner les features valides
feature_cols = [col for col in train_features.columns 
                if col not in ['Date', 'Market_Regime']]
X = train_features[feature_cols]
X_test = test_features[feature_cols]

# Nettoyer
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

# Supprimer les colonnes avec trop de NaN
valid_cols = X.columns[X.isnull().sum() < len(X) * 0.5]
X = X[valid_cols]
X_test = X_test[valid_cols]

# Supprimer les premières lignes
min_valid_row = 60
X = X.iloc[min_valid_row:].reset_index(drop=True)
y = y[min_valid_row:]

print(f"  Shape finale: {X.shape}")

# 5. SÉLECTION PAR IMPORTANCE
print("\n[5] Calcul de l'importance des features...")

# Random Forest pour l'importance
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

# Top features
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Sélectionner les meilleures features
n_features = min(300, len(feature_importances))
top_features = feature_importances.head(n_features)['feature'].tolist()

# Toujours inclure certaines features clés
key_features = [col for col in X.columns if any(x in col for x in ['hmm_', 'turbulence', 'avg_corr', 'wavelet'])]
selected_features = list(set(top_features + key_features))[:300]

X_selected = X[selected_features]
X_test_selected = X_test[selected_features]

print(f"  Features sélectionnées: {len(selected_features)}")

# 6. MODÈLES OPTIMISÉS
print("\n[6] Configuration des modèles...")

models = {
    'xgb': xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1
    ),
    
    'lgb': lgb.LGBMClassifier(
        n_estimators=450,
        num_leaves=31,
        learning_rate=0.03,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    ),
    
    'cat': CatBoostClassifier(
        iterations=400,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=5,
        random_state=42,
        verbose=False
    ),
    
    'extra': ExtraTreesClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
}

# 7. VALIDATION CROISÉE STRATIFIÉE
print("\n[7] Validation croisée...")

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_predictions = {name: np.zeros((len(X_selected), 3)) for name in models}
test_predictions = {name: np.zeros((len(X_test_selected), 3)) for name in models}

for name, model in models.items():
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

# 8. ENSEMBLE INTELLIGENT
print("\n[8] Ensemble des modèles...")

# Poids optimaux basés sur les performances OOF
weights = {}
for name in models:
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

# 9. CALIBRATION ET POST-PROCESSING
print("\n[9] Calibration et post-processing...")

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

# 10. AJUSTEMENT FINAL
print("\n[10] Ajustement final...")

# Distribution cible
target_dist = {-1: 0.272, 0: 0.363, 1: 0.365}

# Ajustement progressif
adjusted_proba = calibrated_test.copy()
for strength in [0.1, 0.2, 0.3, 0.4, 0.5]:
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

# Prédictions finales avec logique de marché
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
            if not pd.isna(turb) and turb > train_features['turbulence_index'].quantile(0.85):
                proba[1] *= 0.85
    
    # Renormaliser
    proba = proba / proba.sum()
    
    # Persistance
    if i > 0 and len(final_predictions) > 0:
        if max(proba) < 0.45:
            prev_pred = final_predictions[-1]
            proba[prev_pred] *= 1.3
            proba = proba / proba.sum()
    
    pred = np.argmax(proba)
    final_predictions.append(pred)

# Décoder
final_predictions = label_encoder.inverse_transform(final_predictions)

# 11. ANALYSE ET SOUMISSION
print("\n[11] Création de la soumission...")

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

submission.to_csv('submission_ultimate_robust.csv', index=False)

print("\n" + "="*80)
print("MODÈLE ULTIMATE ROBUSTE TERMINÉ!")
print("\nCaractéristiques:")
print("  ✓ 300+ features avancées (wavelets, microstructure, HMM)")
print("  ✓ Ensemble de 4 modèles avec poids optimaux")
print("  ✓ Calibration isotonique")
print("  ✓ Post-processing intelligent")
print(f"  ✓ Performance: AUC {final_oof_score:.4f}")
print("="*80)