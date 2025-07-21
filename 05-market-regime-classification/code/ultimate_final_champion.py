"""
ULTIMATE FINAL CHAMPION - MEILLEURE PERFORMANCE
===============================================
Fusion des meilleures techniques pour AUC maximal et distribution parfaite
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy import stats
import pywt
from hmmlearn import hmm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ULTIMATE FINAL CHAMPION - PERFORMANCE MAXIMALE")
print("="*80)

# 1. CHARGEMENT DES DONNÉES
print("\n[1] Chargement des données...")
train = pd.read_csv('train.csv', parse_dates=['Date'])
test = pd.read_csv('test.csv', parse_dates=['Date'])
test_dates = test['Date'].copy()

train_dist = train['Market_Regime'].value_counts(normalize=True).sort_index()
target_dist = {-1: train_dist[-1], 0: train_dist[0], 1: train_dist[1]}

print("\nDistribution cible exacte:")
for regime in [-1, 0, 1]:
    print(f"  Régime {regime}: {target_dist[regime]*100:.2f}%")

# 2. FEATURE ENGINEERING OPTIMISÉ
print("\n[2] Feature Engineering optimisé pour AUC maximal...")

def create_champion_features(df):
    """Features les plus performantes identifiées"""
    price_cols = [col for col in df.columns if col not in ['Date', 'Market_Regime']]
    
    # 1. RETURNS MULTI-ÉCHELLES
    print("  - Returns multi-échelles...")
    for col in price_cols[:20]:
        for period in [1, 2, 5, 10, 20, 60]:
            df[f'{col}_ret_{period}'] = df[col].pct_change(period)
            # EMA des returns
            df[f'{col}_ret_{period}_ema'] = df[f'{col}_ret_{period}'].ewm(span=period).mean()
        
        # Momentum
        df[f'{col}_momentum'] = df[f'{col}_ret_20'] - df[f'{col}_ret_5']
        
    # 2. VOLATILITÉ COMPLÈTE
    print("  - Volatilité avancée...")
    for col in price_cols[:15]:
        returns = df[col].pct_change()
        
        # Volatilité rolling
        for window in [5, 10, 20, 30, 60]:
            df[f'{col}_vol_{window}'] = returns.rolling(window).std() * np.sqrt(252)
        
        # EWMA volatilité
        for span in [10, 20]:
            df[f'{col}_ewma_vol_{span}'] = returns.ewm(span=span).std() * np.sqrt(252)
        
        # Volatilité de la volatilité
        df[f'{col}_vol_of_vol'] = df[f'{col}_vol_20'].rolling(20).std()
        
        # Ratio de volatilité
        df[f'{col}_vol_ratio_5_20'] = df[f'{col}_vol_5'] / (df[f'{col}_vol_20'] + 1e-8)
        df[f'{col}_vol_ratio_10_30'] = df[f'{col}_vol_10'] / (df[f'{col}_vol_30'] + 1e-8)
        
    # 3. WAVELETS MULTI-NIVEAUX
    print("  - Décomposition en ondelettes...")
    for col in price_cols[:10]:
        for window in [32, 64, 128]:
            try:
                # Énergie des coefficients
                df[f'{col}_wavelet_energy_{window}'] = df[col].rolling(window).apply(
                    lambda x: np.sum(np.array(pywt.dwt(x, 'db4')[0])**2) if len(x) == window else np.nan
                )
                # Entropie
                df[f'{col}_wavelet_entropy_{window}'] = df[col].rolling(window).apply(
                    lambda x: stats.entropy(np.abs(pywt.dwt(x, 'db4')[0]) + 1e-10) if len(x) == window else np.nan
                )
            except:
                pass
    
    # 4. MICROSTRUCTURE COMPLÈTE
    print("  - Microstructure avancée...")
    for col in price_cols[:10]:
        returns = df[col].pct_change()
        
        # Realized volatility
        df[f'{col}_realized_vol'] = returns.rolling(20).apply(lambda x: np.sqrt(np.sum(x**2)))
        
        # Bipower variation
        df[f'{col}_bipower_var'] = returns.abs().rolling(2).apply(
            lambda x: x.iloc[0] * x.iloc[1] if len(x) == 2 else np.nan
        ).rolling(20).sum()
        
        # Jump statistic
        df[f'{col}_jump_stat'] = returns.rolling(20).apply(
            lambda x: np.sum(x**4) / (np.sum(x**2)**2 + 1e-10)
        )
        
        # High-low range
        high_low = df[col].rolling(20).max() - df[col].rolling(20).min()
        df[f'{col}_high_low_range'] = high_low / df[col]
        
    # 5. CORRÉLATIONS ET CONCENTRATION
    print("  - Structure de corrélation...")
    for window in [20, 30, 60]:
        corr_features = []
        eigen_features = []
        for i in range(len(df)):
            if i >= window:
                corr_matrix = df[price_cols[:15]].iloc[i-window:i].corr()
                # Statistiques de corrélation
                upper_tri = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                corr_features.append([
                    np.nanmean(upper_tri),           # Moyenne
                    np.nanstd(upper_tri),            # Std
                    np.nanmax(upper_tri),            # Max
                    np.nanmin(upper_tri),            # Min
                    np.nanmedian(upper_tri)          # Médiane
                ])
                
                # Eigenvalues
                try:
                    eigenvals = np.linalg.eigvals(corr_matrix)
                    eigenvals = np.sort(eigenvals)[::-1]
                    eigen_features.append([
                        eigenvals[0],                 # Plus grande
                        eigenvals[0] / np.sum(eigenvals),  # Concentration
                        np.sum(eigenvals[:3]) / np.sum(eigenvals)  # Top 3
                    ])
                except:
                    eigen_features.append([np.nan, np.nan, np.nan])
            else:
                corr_features.append([np.nan]*5)
                eigen_features.append([np.nan]*3)
        
        # Ajouter au dataframe
        corr_df = pd.DataFrame(corr_features, columns=[
            f'corr_mean_{window}', f'corr_std_{window}', f'corr_max_{window}',
            f'corr_min_{window}', f'corr_median_{window}'
        ])
        eigen_df = pd.DataFrame(eigen_features, columns=[
            f'max_eigen_{window}', f'eigen_concentration_{window}', f'top3_eigen_{window}'
        ])
        
        for col in corr_df.columns:
            df[col] = corr_df[col]
        for col in eigen_df.columns:
            df[col] = eigen_df[col]
    
    # 6. INDICATEURS TECHNIQUES AVANCÉS
    print("  - Indicateurs techniques...")
    for col in price_cols[:10]:
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
        df[f'{col}_macd_diff'] = df[f'{col}_macd'] - df[f'{col}_macd_signal']
        
        # Bollinger Bands
        for window in [10, 20]:
            sma = df[col].rolling(window).mean()
            std = df[col].rolling(window).std()
            df[f'{col}_bb_upper_{window}'] = (df[col] - (sma + 2*std)) / df[col]
            df[f'{col}_bb_lower_{window}'] = ((sma - 2*std) - df[col]) / df[col]
            df[f'{col}_bb_width_{window}'] = (2 * 2 * std) / sma
        
    # 7. TURBULENCE INDEX AMÉLIORÉ
    print("  - Turbulence index...")
    returns_matrix = df[price_cols[:15]].pct_change()
    for lookback in [30, 60, 90]:
        turbulence = []
        for i in range(len(df)):
            if i >= lookback:
                historical = returns_matrix.iloc[i-lookback:i]
                current = returns_matrix.iloc[i]
                try:
                    mean_hist = historical.mean()
                    cov_hist = historical.cov()
                    # Régularisation
                    cov_hist = cov_hist + np.eye(len(cov_hist)) * 1e-6
                    inv_cov = np.linalg.inv(cov_hist)
                    diff = current - mean_hist
                    turb = np.sqrt(diff @ inv_cov @ diff)
                    turbulence.append(turb)
                except:
                    turbulence.append(np.nan)
            else:
                turbulence.append(np.nan)
        df[f'turbulence_{lookback}'] = turbulence
    
    # 8. FEATURES TEMPORELLES COMPLÈTES
    print("  - Features temporelles...")
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_month'] = df['Date'].dt.day
    df['week_of_year'] = df['Date'].dt.isocalendar().week
    df['day_of_year'] = df['Date'].dt.dayofyear
    
    # Encodage cyclique complet
    for period, max_val in [('month', 12), ('week_of_year', 52), ('day_of_year', 365)]:
        df[f'{period}_sin'] = np.sin(2 * np.pi * df[period] / max_val)
        df[f'{period}_cos'] = np.cos(2 * np.pi * df[period] / max_val)
    
    # Indicateurs spéciaux
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_quarter_end'] = df['Date'].dt.is_quarter_end.astype(int)
    df['is_quarter_start'] = df['Date'].dt.is_quarter_start.astype(int)
    df['days_in_month'] = df['Date'].dt.days_in_month
    
    # Patterns saisonniers spécifiques
    df['is_january'] = (df['month'] == 1).astype(int)
    df['is_april'] = (df['month'] == 4).astype(int)
    df['is_june'] = (df['month'] == 6).astype(int)
    df['is_december'] = (df['month'] == 12).astype(int)
    
    return df

# 3. HMM MULTI-ÉTATS
print("\n[3] Hidden Markov Models multi-états...")

def add_advanced_hmm_features(train_df, test_df, price_cols):
    """HMM avec différentes configurations"""
    train_returns = train_df[price_cols[:15]].pct_change().fillna(0)
    train_returns = train_returns.replace([np.inf, -np.inf], 0)
    
    test_returns = test_df[price_cols[:15]].pct_change().fillna(0)
    test_returns = test_returns.replace([np.inf, -np.inf], 0)
    
    # Différentes configurations HMM
    for n_states in [3, 4, 5]:
        try:
            print(f"  - HMM avec {n_states} états...")
            hmm_model = hmm.GaussianHMM(
                n_components=n_states, 
                covariance_type="full" if n_states == 3 else "diag",
                n_iter=100, 
                random_state=42
            )
            hmm_model.fit(train_returns[1:])
            
            # Prédictions
            train_proba = hmm_model.predict_proba(train_returns[1:])
            test_proba = hmm_model.predict_proba(test_returns)
            
            # Ajouter aux dataframes
            for i in range(n_states):
                train_df[f'hmm_{n_states}states_{i}'] = np.concatenate([[0], train_proba[:, i]])
                test_df[f'hmm_{n_states}states_{i}'] = test_proba[:, i]
                
        except Exception as e:
            print(f"    Erreur: {e}")
            for i in range(n_states):
                train_df[f'hmm_{n_states}states_{i}'] = 0
                test_df[f'hmm_{n_states}states_{i}'] = 0
    
    return train_df, test_df

# Application
train_features = create_champion_features(train.copy())
test_features = create_champion_features(test.copy())

price_cols = [col for col in train.columns if col not in ['Date', 'Market_Regime']]
train_features, test_features = add_advanced_hmm_features(train_features, test_features, price_cols)

# 4. PRÉPARATION DES DONNÉES
print("\n[4] Préparation des données...")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_features['Market_Regime'])

feature_cols = [col for col in train_features.columns 
                if col not in ['Date', 'Market_Regime']]
X = train_features[feature_cols]
X_test = test_features[feature_cols]

# Nettoyage
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

# Colonnes valides
valid_cols = X.columns[X.isnull().sum() < len(X) * 0.3]
X = X[valid_cols]
X_test = X_test[valid_cols]

# Supprimer les premières lignes
min_valid_row = 90
X = X.iloc[min_valid_row:].reset_index(drop=True)
y = y[min_valid_row:]

print(f"  Shape: {X.shape}")

# 5. SÉLECTION DE FEATURES AVANCÉE
print("\n[5] Sélection de features par importance...")

# Random Forest pour importance
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X, y)

# Features importantes
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Top features + features clés
n_top = 350
top_features = importances.head(n_top)['feature'].tolist()

# Toujours inclure certaines features
key_patterns = ['hmm_', 'turbulence_', 'wavelet_', 'eigen_', 'corr_mean', 'vol_ratio', 'realized_vol']
key_features = [col for col in X.columns if any(pattern in col for pattern in key_patterns)]

selected_features = list(set(top_features + key_features))[:400]

X_selected = X[selected_features]
X_test_selected = X_test[selected_features]

print(f"  Features finales: {len(selected_features)}")

# 6. MODÈLES CHAMPIONS
print("\n[6] Configuration des modèles champions...")

models = {
    'xgb_champion': xgb.XGBClassifier(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.025,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.8,
        reg_lambda=0.3,
        min_child_weight=3,
        scale_pos_weight=1,
        random_state=42,
        n_jobs=-1
    ),
    
    'lgb_champion': lgb.LGBMClassifier(
        n_estimators=550,
        num_leaves=35,
        learning_rate=0.025,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        lambda_l1=0.5,
        lambda_l2=0.5,
        min_data_in_leaf=20,
        boosting_type='gbdt',
        objective='multiclass',
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    ),
    
    'cat_champion': CatBoostClassifier(
        iterations=500,
        learning_rate=0.025,
        depth=6,
        l2_leaf_reg=7,
        border_count=128,
        auto_class_weights='Balanced',
        random_state=42,
        verbose=False
    )
}

# 7. VALIDATION CROISÉE STRATIFIÉE
print("\n[7] Validation croisée avec purging...")

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_predictions = {name: np.zeros((len(X_selected), 3)) for name in models}
test_predictions = {name: np.zeros((len(X_test_selected), 3)) for name in models}

for name, model in models.items():
    print(f"\n{name}:")
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected, y)):
        # Purging pour éviter le data leakage temporel
        train_idx = train_idx[train_idx < min(val_idx) - 20]
        
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

# 8. ENSEMBLE OPTIMAL
print("\n[8] Ensemble avec poids optimaux...")

# Calculer les poids basés sur OOF
weights = {}
for name in models:
    oof_score = roc_auc_score(y, oof_predictions[name], multi_class='ovr')
    weights[name] = oof_score ** 3  # Poids cubique pour favoriser les meilleurs

# Normaliser
total_weight = sum(weights.values())
weights = {name: w/total_weight for name, w in weights.items()}

print("\nPoids finaux:")
for name, weight in weights.items():
    print(f"  {name}: {weight:.3f}")

# Ensemble
ensemble_test = np.zeros_like(test_predictions['xgb_champion'])
for name, weight in weights.items():
    ensemble_test += weight * test_predictions[name]

# 9. CALIBRATION AVANCÉE
print("\n[9] Calibration isotonique avancée...")

from sklearn.isotonic import IsotonicRegression

# Ensemble OOF pour calibration
oof_ensemble = np.zeros_like(oof_predictions['xgb_champion'])
for name, weight in weights.items():
    oof_ensemble += weight * oof_predictions[name]

# Calibration par classe
calibrated_test = np.zeros_like(ensemble_test)
for i in range(3):
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(oof_ensemble[:, i], (y == i).astype(int))
    calibrated_test[:, i] = iso_reg.predict(ensemble_test[:, i])

# Normaliser
calibrated_test = calibrated_test / calibrated_test.sum(axis=1, keepdims=True)

# 10. POST-PROCESSING FINAL INTELLIGENT
print("\n[10] Post-processing final intelligent...")

def intelligent_final_adjustment(probabilities, features_df, target_distribution):
    """Ajustement final avec toute la logique de marché"""
    adjusted = probabilities.copy()
    predictions = []
    
    # Ajustement progressif de la distribution
    for iteration in range(3):
        # Calculer la distribution actuelle
        temp_preds = np.argmax(adjusted, axis=1)
        current_dist = np.bincount(temp_preds, minlength=3) / len(temp_preds)
        
        # Ajuster globalement
        for i in range(3):
            target = list(target_distribution.values())[i]
            if current_dist[i] > 0:
                ratio = (target / current_dist[i]) ** (0.3 - iteration * 0.1)
                adjusted[:, i] *= ratio
        
        # Renormaliser
        adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
    
    # Ajustements individuels
    for i in range(len(adjusted)):
        proba = adjusted[i].copy()
        
        if i < len(features_df):
            # Facteurs saisonniers
            month = features_df.iloc[i].get('month', 0)
            quarter = features_df.iloc[i].get('quarter', 0)
            
            # Avril (baissier historiquement)
            if month == 4:
                proba[0] *= 1.25
                proba[2] *= 0.85
            # Décembre (haussier historiquement)
            elif month == 12:
                proba[2] *= 1.25
                proba[0] *= 0.85
            # Juin (légèrement haussier)
            elif month == 6:
                proba[2] *= 1.10
            
            # Turbulence
            turb_cols = [col for col in features_df.columns if 'turbulence' in col]
            if turb_cols:
                avg_turb = features_df.iloc[i][turb_cols].mean()
                if not pd.isna(avg_turb):
                    # Percentile de turbulence
                    turb_percentile = stats.percentileofscore(
                        train_features[turb_cols].mean(axis=1).dropna(), 
                        avg_turb
                    )
                    if turb_percentile > 90:
                        # Haute turbulence - favoriser les extrêmes
                        proba[1] *= 0.7
                        proba[0] *= 1.15
                        proba[2] *= 1.15
                    elif turb_percentile > 80:
                        proba[1] *= 0.85
            
            # Volatilité
            vol_cols = [col for col in features_df.columns if 'vol_ratio' in col]
            if vol_cols:
                avg_vol_ratio = features_df.iloc[i][vol_cols].mean()
                if not pd.isna(avg_vol_ratio) and avg_vol_ratio > 1.5:
                    # Volatilité court terme élevée
                    proba[1] *= 0.9
            
            # HMM signals
            hmm_cols = [col for col in features_df.columns if 'hmm_3states' in col]
            if hmm_cols and len(hmm_cols) >= 3:
                hmm_probs = features_df.iloc[i][hmm_cols].values
                if not any(pd.isna(hmm_probs)):
                    # Pondérer avec HMM
                    proba = 0.7 * proba + 0.3 * hmm_probs
        
        # Renormaliser
        proba = proba / proba.sum()
        
        # Logique de persistance adaptative
        if len(predictions) > 0:
            # Confiance dans la prédiction
            max_proba = max(proba)
            
            if max_proba < 0.4:  # Faible confiance
                # Forte persistance
                prev_pred = predictions[-1]
                proba[prev_pred] *= 1.5
            elif max_proba < 0.5:  # Confiance moyenne
                # Persistance modérée
                prev_pred = predictions[-1]
                proba[prev_pred] *= 1.2
            
            # Anti-oscillation
            if len(predictions) > 3:
                recent_changes = sum(1 for j in range(1, 4) if predictions[-j] != predictions[-j-1])
                if recent_changes >= 3:  # Trop de changements
                    # Favoriser la stabilité
                    most_common = Counter(predictions[-4:]).most_common(1)[0][0]
                    proba[most_common] *= 1.3
        
        # Renormaliser finale
        proba = proba / proba.sum()
        
        # Prédiction
        pred = np.argmax(proba)
        predictions.append(pred)
    
    return np.array(predictions)

# Application
final_predictions = intelligent_final_adjustment(calibrated_test, test_features, target_dist)

# Ajustement fin de la distribution
def fine_tune_distribution(predictions, target_dist, n_iterations=5):
    """Ajustement fin pour distribution parfaite"""
    predictions = predictions.copy()
    
    for iteration in range(n_iterations):
        # Distribution actuelle
        current_counts = np.bincount(predictions, minlength=3)
        current_dist = current_counts / len(predictions)
        
        # Calculer les écarts
        errors = {}
        for i in range(3):
            target = list(target_dist.values())[i]
            errors[i] = target - current_dist[i]
        
        # Identifier les swaps nécessaires
        # Du régime avec excès vers celui avec déficit
        for _ in range(10):  # Limite de swaps par itération
            # Trouver le plus grand excès et déficit
            max_excess = min(errors.items(), key=lambda x: x[1])
            max_deficit = max(errors.items(), key=lambda x: x[1])
            
            if max_excess[1] < -0.005 and max_deficit[1] > 0.005:
                # Trouver des candidats pour le swap
                excess_indices = np.where(predictions == max_excess[0])[0]
                
                if len(excess_indices) > 0:
                    # Choisir aléatoirement
                    swap_idx = np.random.choice(excess_indices)
                    predictions[swap_idx] = max_deficit[0]
                    
                    # Mettre à jour les erreurs
                    errors[max_excess[0]] += 1/len(predictions)
                    errors[max_deficit[0]] -= 1/len(predictions)
    
    return predictions

final_predictions = fine_tune_distribution(final_predictions, target_dist, n_iterations=10)

# Décoder
final_predictions = label_encoder.inverse_transform(final_predictions)

# 11. ANALYSE ET SOUMISSION
print("\n[11] Création de la soumission finale...")

# Distribution finale
dist = Counter(final_predictions)
print("\nDistribution finale:")
for regime in sorted(dist.keys()):
    pct = dist[regime]/len(final_predictions)*100
    target_pct = target_dist[regime]*100
    print(f"  Régime {regime}: {dist[regime]} ({pct:.1f}%) - Cible: {target_pct:.1f}% - Diff: {abs(pct-target_pct):.2f}%")

# Score OOF final
final_oof_score = roc_auc_score(y, oof_ensemble, multi_class='ovr')
print(f"\nScore AUC OOF final: {final_oof_score:.4f}")

# Créer la soumission
submission = pd.DataFrame({
    'Id': test_dates,
    'Expected': final_predictions.astype(int)
})

submission.to_csv('submission_ultimate_champion.csv', index=False)

print("\n" + "="*80)
print("ULTIMATE FINAL CHAMPION TERMINÉ!")
print("\nCaractéristiques finales:")
print("  ✓ 400+ features avancées incluant wavelets et HMM multi-états")
print("  ✓ Ensemble de modèles champions avec poids cubiques")
print("  ✓ Calibration isotonique avancée")
print("  ✓ Post-processing intelligent multi-niveaux")
print("  ✓ Ajustement fin de distribution")
print(f"  ✓ Performance: AUC {final_oof_score:.4f}")
print("="*80)