"""
ULTIMATE OPTIMIZED FINAL - MEILLEUR MODÈLE STABLE
=================================================
Combinaison optimale de toutes les techniques qui fonctionnent
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                             HistGradientBoostingClassifier, VotingClassifier)
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from hmmlearn import hmm
from sklearn.mixture import GaussianMixture
from collections import Counter
from scipy import stats
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ULTIMATE OPTIMIZED FINAL - PERFORMANCE MAXIMALE")
print("="*80)

# 1. CHARGEMENT DES DONNÉES
print("\n[1] Chargement des données...")
train = pd.read_csv('train.csv', parse_dates=['Date'])
test = pd.read_csv('test.csv', parse_dates=['Date'])
test_dates = test['Date'].copy()

target_dist = train['Market_Regime'].value_counts(normalize=True).sort_index().to_dict()
print("\nDistribution cible:")
for regime, pct in target_dist.items():
    print(f"  Régime {regime}: {pct*100:.2f}%")

# 2. FEATURE ENGINEERING COMPLET
print("\n[2] Feature Engineering complet...")

def create_ultimate_features(df):
    """Toutes les meilleures features identifiées"""
    price_cols = [col for col in df.columns if col not in ['Date', 'Market_Regime']]
    
    # 1. RETURNS ET TRANSFORMATIONS
    print("  - Returns et transformations...")
    for col in price_cols[:25]:
        # Returns classiques
        for period in [1, 2, 5, 10, 20, 60]:
            df[f'{col}_ret_{period}'] = df[col].pct_change(period)
            
        # Log returns
        df[f'{col}_log_ret'] = np.log(df[col] / df[col].shift(1))
        
        # Returns normalisés
        ret = df[col].pct_change()
        df[f'{col}_ret_zscore'] = (ret - ret.rolling(60).mean()) / ret.rolling(60).std()
        
    # 2. VOLATILITÉ COMPLÈTE
    print("  - Volatilité complète...")
    for col in price_cols[:20]:
        returns = df[col].pct_change()
        
        # Volatilité simple
        for window in [5, 10, 20, 30, 60]:
            df[f'{col}_vol_{window}'] = returns.rolling(window).std() * np.sqrt(252)
        
        # EWMA volatilité
        df[f'{col}_ewma_vol'] = returns.ewm(span=20).std() * np.sqrt(252)
        
        # Volatilité réalisée
        df[f'{col}_realized_vol'] = returns.rolling(20).apply(lambda x: np.sqrt(np.sum(x**2)))
        
        # Volatilité de la volatilité
        df[f'{col}_vol_of_vol'] = df[f'{col}_vol_20'].rolling(20).std()
        
        # Ratio de volatilité
        df[f'{col}_vol_ratio'] = df[f'{col}_vol_5'] / (df[f'{col}_vol_20'] + 1e-8)
        
    # 3. MOMENTS STATISTIQUES
    print("  - Moments statistiques...")
    for col in price_cols[:15]:
        returns = df[col].pct_change()
        
        for window in [10, 20, 30]:
            # Skewness et Kurtosis
            df[f'{col}_skew_{window}'] = returns.rolling(window).skew()
            df[f'{col}_kurt_{window}'] = returns.rolling(window).kurt()
            
            # Jarque-Bera test
            df[f'{col}_jb_{window}'] = returns.rolling(window).apply(
                lambda x: stats.jarque_bera(x)[0] if len(x) == window and not np.isnan(x).any() else np.nan
            )
    
    # 4. INDICATEURS TECHNIQUES
    print("  - Indicateurs techniques...")
    for col in price_cols[:15]:
        # RSI multiple périodes
        for period in [7, 14, 21]:
            delta = df[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-8)
            df[f'{col}_rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df[col].ewm(span=12).mean()
        ema26 = df[col].ewm(span=26).mean()
        df[f'{col}_macd'] = ema12 - ema26
        df[f'{col}_macd_signal'] = df[f'{col}_macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        for window in [10, 20]:
            sma = df[col].rolling(window).mean()
            std = df[col].rolling(window).std()
            df[f'{col}_bb_upper_{window}'] = (df[col] - (sma + 2*std)) / df[col]
            df[f'{col}_bb_lower_{window}'] = ((sma - 2*std) - df[col]) / df[col]
            df[f'{col}_bb_width_{window}'] = 4 * std / sma
        
        # Stochastic
        for period in [14]:
            low_min = df[col].rolling(period).min()
            high_max = df[col].rolling(period).max()
            df[f'{col}_stoch_{period}'] = 100 * (df[col] - low_min) / (high_max - low_min + 1e-8)
    
    # 5. CORRÉLATIONS ET STRUCTURE
    print("  - Corrélations et structure...")
    for window in [20, 30, 60]:
        corr_features = []
        for i in range(len(df)):
            if i >= window:
                corr_matrix = df[price_cols[:15]].iloc[i-window:i].corr()
                upper_tri = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                
                # Statistiques de corrélation
                avg_corr = np.nanmean(upper_tri)
                std_corr = np.nanstd(upper_tri)
                max_corr = np.nanmax(upper_tri) if len(upper_tri) > 0 else np.nan
                
                # Eigenvalues
                try:
                    eigenvals = np.linalg.eigvals(corr_matrix)
                    max_eigen = np.max(eigenvals)
                    eigen_ratio = max_eigen / np.sum(eigenvals)
                except:
                    max_eigen = np.nan
                    eigen_ratio = np.nan
                
                corr_features.append([avg_corr, std_corr, max_corr, max_eigen, eigen_ratio])
            else:
                corr_features.append([np.nan] * 5)
        
        corr_df = pd.DataFrame(corr_features, columns=[
            f'corr_avg_{window}', f'corr_std_{window}', f'corr_max_{window}',
            f'eigen_max_{window}', f'eigen_ratio_{window}'
        ])
        
        for col in corr_df.columns:
            df[col] = corr_df[col]
    
    # 6. TURBULENCE INDEX
    print("  - Turbulence index...")
    returns_matrix = df[price_cols[:15]].pct_change()
    for lookback in [30, 60]:
        turbulence = []
        for i in range(len(df)):
            if i >= lookback:
                historical = returns_matrix.iloc[i-lookback:i]
                current = returns_matrix.iloc[i]
                try:
                    mean_hist = historical.mean()
                    cov_hist = historical.cov() + np.eye(len(historical.columns)) * 1e-6
                    inv_cov = np.linalg.inv(cov_hist)
                    diff = current - mean_hist
                    turb = np.sqrt(diff @ inv_cov @ diff)
                    turbulence.append(turb)
                except:
                    turbulence.append(np.nan)
            else:
                turbulence.append(np.nan)
        df[f'turbulence_{lookback}'] = turbulence
    
    # 7. HMM ET GMM
    print("  - Modèles de régime HMM/GMM...")
    returns_data = df[price_cols[:15]].pct_change().fillna(0)
    returns_data = returns_data.replace([np.inf, -np.inf], 0)
    
    # HMM avec différents états
    for n_states in [3, 4]:
        try:
            hmm_model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", 
                                       n_iter=100, random_state=42)
            hmm_model.fit(returns_data[1:])
            proba = hmm_model.predict_proba(returns_data[1:])
            
            for i in range(n_states):
                df[f'hmm_{n_states}states_{i}'] = np.concatenate([[0], proba[:, i]])
        except:
            for i in range(n_states):
                df[f'hmm_{n_states}states_{i}'] = 0
    
    # GMM
    try:
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(returns_data[1:])
        proba = gmm.predict_proba(returns_data[1:])
        
        for i in range(3):
            df[f'gmm_3_{i}'] = np.concatenate([[0], proba[:, i]])
    except:
        for i in range(3):
            df[f'gmm_3_{i}'] = 0
    
    # 8. FEATURES TEMPORELLES
    print("  - Features temporelles...")
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_month'] = df['Date'].dt.day
    df['week_of_year'] = df['Date'].dt.isocalendar().week
    df['day_of_year'] = df['Date'].dt.dayofyear
    
    # Encodage cyclique
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
    
    # Indicateurs spéciaux
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_quarter_end'] = df['Date'].dt.is_quarter_end.astype(int)
    df['is_january'] = (df['month'] == 1).astype(int)
    df['is_april'] = (df['month'] == 4).astype(int)
    df['is_june'] = (df['month'] == 6).astype(int)
    df['is_december'] = (df['month'] == 12).astype(int)
    
    return df

# Application
train_features = create_ultimate_features(train.copy())
test_features = create_ultimate_features(test.copy())

# 3. PRÉPARATION DES DONNÉES
print("\n[3] Préparation des données...")

# Labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_features['Market_Regime'])

# Features
feature_cols = [col for col in train_features.columns if col not in ['Date', 'Market_Regime']]
X = train_features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = test_features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

# Remove initial rows
min_valid_row = 100
X = X.iloc[min_valid_row:].reset_index(drop=True)
y = y[min_valid_row:]

# Remove low variance features
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.001)
X_selected = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])
X_test_selected = pd.DataFrame(selector.transform(X_test), columns=X.columns[selector.get_support()])

print(f"Features après sélection: {X_selected.shape[1]}")

# 4. SÉLECTION PAR IMPORTANCE
print("\n[4] Sélection des meilleures features...")

# Random Forest pour l'importance
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_selected, y)

# Top features
importances = pd.DataFrame({
    'feature': X_selected.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Sélectionner les meilleures
n_features = min(400, len(importances))
top_features = importances.head(n_features)['feature'].tolist()

# Toujours inclure les features clés
key_patterns = ['hmm_', 'gmm_', 'turbulence_', 'eigen_', 'corr_avg', 'vol_ratio', 'realized_vol']
key_features = [col for col in X_selected.columns if any(pattern in col for pattern in key_patterns)]
final_features = list(set(top_features + key_features))[:400]

X_final = X_selected[final_features]
X_test_final = X_test_selected[final_features]

# Scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_final)
X_test_scaled = scaler.transform(X_test_final)

print(f"Shape finale: {X_scaled.shape}")

# 5. MODÈLES OPTIMISÉS
print("\n[5] Configuration des modèles optimisés...")

models = {
    'xgb_final': xgb.XGBClassifier(
        n_estimators=700,
        max_depth=7,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=1.0,
        reg_lambda=0.5,
        scale_pos_weight=1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    ),
    
    'lgb_final': lgb.LGBMClassifier(
        n_estimators=650,
        num_leaves=40,
        learning_rate=0.025,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        lambda_l1=0.5,
        lambda_l2=0.5,
        min_data_in_leaf=20,
        random_state=42,
        verbosity=-1
    ),
    
    'cat_final': CatBoostClassifier(
        iterations=600,
        learning_rate=0.025,
        depth=6,
        l2_leaf_reg=7,
        border_count=128,
        auto_class_weights='Balanced',
        random_state=42,
        verbose=False
    ),
    
    'hist_gb': HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_depth=8,
        l2_regularization=1.0,
        random_state=42
    ),
    
    'extra_trees': ExtraTreesClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    ),
    
    'neural_net': MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        alpha=0.01,
        batch_size=32,
        learning_rate='adaptive',
        max_iter=300,
        random_state=42
    )
}

# 6. VALIDATION STRATIFIÉE
print("\n[6] Validation croisée stratifiée...")

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_predictions = {}
all_scores = {}
oof_predictions = {}

for name, model in models.items():
    print(f"\n{name}:")
    scores = []
    test_preds = []
    oof_pred = np.zeros((len(X_scaled), 3))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        # Purging temporel
        if fold > 0:
            train_idx = train_idx[train_idx < min(val_idx) - 20]
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Clone et entraîner
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train, y_train)
        
        # Prédictions
        val_pred = model_clone.predict_proba(X_val)
        oof_pred[val_idx] = val_pred
        
        # Score
        score = roc_auc_score(y_val, val_pred, multi_class='ovr')
        scores.append(score)
        print(f"  Fold {fold+1}: {score:.4f}")
        
        # Test predictions
        test_pred = model_clone.predict_proba(X_test_scaled)
        test_preds.append(test_pred)
    
    print(f"  Moyenne: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    
    # Store results
    all_scores[name] = np.mean(scores)
    all_predictions[name] = np.mean(test_preds, axis=0)
    oof_predictions[name] = oof_pred

# 7. ENSEMBLE OPTIMAL
print("\n[7] Ensemble optimal...")

# Poids basés sur la performance
weights = {}
total_score = sum(all_scores.values())
for name, score in all_scores.items():
    # Poids quadratique pour favoriser les meilleurs
    weights[name] = (score ** 2) / sum(s ** 2 for s in all_scores.values())

print("\nPoids finaux:")
for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {weight:.3f} (AUC: {all_scores[name]:.4f})")

# Ensemble
ensemble_proba = np.zeros_like(all_predictions['xgb_final'])
for name, weight in weights.items():
    ensemble_proba += weight * all_predictions[name]

# OOF ensemble pour calibration
oof_ensemble = np.zeros_like(oof_predictions['xgb_final'])
for name, weight in weights.items():
    oof_ensemble += weight * oof_predictions[name]

# Score OOF final
oof_score = roc_auc_score(y, oof_ensemble, multi_class='ovr')
print(f"\nScore AUC OOF ensemble: {oof_score:.4f}")

# 8. CALIBRATION ISOTONIQUE
print("\n[8] Calibration isotonique...")

from sklearn.isotonic import IsotonicRegression

calibrated_proba = np.zeros_like(ensemble_proba)
for i in range(3):
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(oof_ensemble[:, i], (y == i).astype(int))
    calibrated_proba[:, i] = iso_reg.predict(ensemble_proba[:, i])

# Normaliser
calibrated_proba = calibrated_proba / calibrated_proba.sum(axis=1, keepdims=True)

# 9. POST-PROCESSING INTELLIGENT
print("\n[9] Post-processing intelligent...")

final_predictions = []
for i in range(len(calibrated_proba)):
    proba = calibrated_proba[i].copy()
    
    # Logique de marché
    if i < len(test_features):
        month = test_features.iloc[i].get('month', 6)
        quarter = test_features.iloc[i].get('quarter', 2)
        
        # Ajustements saisonniers forts
        if month == 4:  # Avril
            proba[0] *= 1.3
            proba[2] *= 0.7
        elif month == 12:  # Décembre
            proba[2] *= 1.3
            proba[0] *= 0.7
        elif month in [1, 2]:  # Effet janvier
            proba[2] *= 1.15
        elif month == 6:  # Juin
            proba[2] *= 1.1
        
        # Turbulence
        turb_cols = [col for col in test_features.columns if 'turbulence' in col]
        if turb_cols:
            avg_turb = test_features.iloc[i][turb_cols].mean()
            if not pd.isna(avg_turb):
                # Percentile dans les données d'entraînement
                turb_values = train_features[turb_cols].mean(axis=1).dropna()
                if len(turb_values) > 0:
                    turb_percentile = stats.percentileofscore(turb_values, avg_turb)
                    if turb_percentile > 90:
                        proba[1] *= 0.6  # Moins de neutre
                        proba[0] *= 1.2
                        proba[2] *= 1.2
    
    # Normaliser
    proba = proba / proba.sum()
    
    # Logique de persistance
    if len(final_predictions) > 0:
        if max(proba) < 0.45:  # Faible confiance
            prev_pred = final_predictions[-1]
            proba[prev_pred] *= 1.5
            proba = proba / proba.sum()
        
        # Anti-oscillation
        if len(final_predictions) > 3:
            recent_changes = sum(1 for j in range(1, 4) if final_predictions[-j] != final_predictions[-j-1])
            if recent_changes >= 3:
                most_common = Counter(final_predictions[-4:]).most_common(1)[0][0]
                proba[most_common] *= 1.4
                proba = proba / proba.sum()
    
    pred = np.argmax(proba)
    final_predictions.append(pred)

# 10. AJUSTEMENT FINAL DE DISTRIBUTION
print("\n[10] Ajustement final de distribution...")

def optimize_distribution(predictions, target_dist, iterations=30):
    predictions = np.array(predictions)
    
    for _ in range(iterations):
        current_counts = np.bincount(predictions, minlength=3)
        current_dist = current_counts / len(predictions)
        
        # Calculer les erreurs
        errors = {}
        for i in range(3):
            errors[i] = list(target_dist.values())[i] - current_dist[i]
        
        # Identifier les swaps nécessaires
        for _ in range(5):  # Limite de swaps par itération
            # Plus grand déficit et excès
            max_deficit = max(errors.items(), key=lambda x: x[1])
            max_excess = min(errors.items(), key=lambda x: x[1])
            
            if max_deficit[1] > 0.005 and max_excess[1] < -0.005:
                # Trouver des candidats
                excess_indices = np.where(predictions == max_excess[0])[0]
                
                if len(excess_indices) > 0:
                    # Choisir basé sur la confiance
                    confidences = calibrated_proba[excess_indices, max_deficit[0]]
                    best_idx = excess_indices[np.argmax(confidences)]
                    
                    # Swap
                    predictions[best_idx] = max_deficit[0]
                    
                    # Mettre à jour les erreurs
                    errors[max_excess[0]] += 1/len(predictions)
                    errors[max_deficit[0]] -= 1/len(predictions)
    
    return predictions

final_predictions = optimize_distribution(final_predictions, target_dist)

# Décoder
final_predictions = label_encoder.inverse_transform(final_predictions)

# 11. SOUMISSION FINALE
print("\n[11] Création de la soumission finale...")

# Distribution finale
dist = Counter(final_predictions)
print("\nDistribution finale:")
for regime in sorted(dist.keys()):
    pct = dist[regime]/len(final_predictions)*100
    target_pct = target_dist[regime]*100
    print(f"  Régime {regime}: {dist[regime]} ({pct:.1f}%) - Cible: {target_pct:.1f}% - Erreur: {abs(pct-target_pct):.2f}%")

# Erreur moyenne
avg_error = np.mean([abs(dist[r]/len(final_predictions) - target_dist[r]) for r in dist.keys()])
print(f"\nErreur de distribution moyenne: {avg_error*100:.2f}%")

# Create submission
submission = pd.DataFrame({
    'Id': test_dates,
    'Expected': final_predictions.astype(int)
})

submission.to_csv('submission_ultimate_optimized.csv', index=False)

print("\n" + "="*80)
print("ULTIMATE OPTIMIZED FINAL TERMINÉ!")
print("\nRésumé:")
print(f"  ✓ {len(final_features)} features sélectionnées")
print(f"  ✓ 6 modèles avec ensemble pondéré")
print(f"  ✓ Score AUC OOF: {oof_score:.4f}")
print(f"  ✓ Erreur de distribution: {avg_error*100:.2f}%")
print("  ✓ Post-processing intelligent appliqué")
print("="*80)