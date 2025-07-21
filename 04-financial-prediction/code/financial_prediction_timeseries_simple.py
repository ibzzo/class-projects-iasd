#!/usr/bin/env python3
"""
Prédiction de Séries Temporelles Financières - Version Simplifiée et Optimisée
Corrige les problèmes identifiés dans la version ultimate

Améliorations clés:
- Feature engineering simplifié pour éviter l'overfitting
- Validation temporelle stricte
- Pas de feature dominante (Feature_38)
- Post-processing intelligent
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from typing import List, Tuple, Dict
import gc
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, HuberRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Set random seeds
np.random.seed(42)

print("🚀 Financial Time Series Prediction - Version Optimisée")
print("=" * 60)

# ========================================
# FONCTIONS UTILITAIRES
# ========================================

def reduce_memory_usage(df):
    """Réduit l'usage mémoire du DataFrame"""
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Ignorer les colonnes non-numériques et datetime
        if col_type == object or np.issubdtype(col_type, np.datetime64):
            continue
            
        c_min = df[col].min()
        c_max = df[col].max()
        
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'  Mémoire réduite de {start_mem:.2f} MB à {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}%)')
    
    return df

# ========================================
# CHARGEMENT DES DONNÉES
# ========================================

def load_data(data_dir='data'):
    """Charge les données avec analyse"""
    print("\n📊 Chargement des données...")
    
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')
    
    # Conversion des dates
    train_df['Dates'] = pd.to_datetime(train_df['Dates'])
    test_df['Dates'] = pd.to_datetime(test_df['Dates'])
    
    # Tri par date - CRUCIAL pour time series
    train_df = train_df.sort_values('Dates').reset_index(drop=True)
    test_df = test_df.sort_values('Dates').reset_index(drop=True)
    
    # Utiliser les dates comme ID
    if 'ID' not in test_df.columns:
        test_df['ID'] = test_df['Dates'].dt.strftime('%Y-%m-%d')
    
    print(f"✅ Données chargées:")
    print(f"   - Train: {train_df.shape}")
    print(f"   - Test: {test_df.shape}")
    print(f"   - Période train: {train_df['Dates'].min()} à {train_df['Dates'].max()}")
    print(f"   - Période test: {test_df['Dates'].min()} à {test_df['Dates'].max()}")
    
    # Analyse de la target
    print(f"\n📈 Analyse de la target:")
    print(f"   - Moyenne: {train_df['ToPredict'].mean():.6f}")
    print(f"   - Std: {train_df['ToPredict'].std():.6f}")
    print(f"   - Min: {train_df['ToPredict'].min():.6f}")
    print(f"   - Max: {train_df['ToPredict'].max():.6f}")
    
    return train_df, test_df

# ========================================
# FEATURE ENGINEERING CONTRÔLÉ
# ========================================

def create_date_features(df):
    """Crée des features temporelles basiques"""
    df = df.copy()
    
    # Features temporelles
    df['year'] = df['Dates'].dt.year
    df['month'] = df['Dates'].dt.month
    df['day'] = df['Dates'].dt.day
    df['dayofweek'] = df['Dates'].dt.dayofweek
    df['quarter'] = df['Dates'].dt.quarter
    df['dayofyear'] = df['Dates'].dt.dayofyear
    df['weekofyear'] = df['Dates'].dt.isocalendar().week
    
    # Features cycliques pour capturer la périodicité
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 30)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 30)
    
    return df

def select_important_features(train_df, feature_cols, target_col='ToPredict', k=30):
    """Sélectionne les features les plus importantes de manière robuste"""
    print(f"\n🔍 Sélection des {k} features les plus importantes...")
    
    # Méthode 1: F-statistics
    selector_f = SelectKBest(f_regression, k=min(k, len(feature_cols)))
    selector_f.fit(train_df[feature_cols], train_df[target_col])
    scores_f = pd.DataFrame({
        'feature': feature_cols,
        'f_score': selector_f.scores_
    }).sort_values('f_score', ascending=False)
    
    # Méthode 2: Mutual Information
    mi_scores = mutual_info_regression(train_df[feature_cols], train_df[target_col], random_state=42)
    scores_mi = pd.DataFrame({
        'feature': feature_cols,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # Méthode 3: RandomForest importance (plus robuste)
    rf = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
    rf.fit(train_df[feature_cols], train_df[target_col])
    scores_rf = pd.DataFrame({
        'feature': feature_cols,
        'rf_importance': rf.feature_importances_
    }).sort_values('rf_importance', ascending=False)
    
    # Combiner les scores
    all_scores = scores_f.merge(scores_mi, on='feature').merge(scores_rf, on='feature')
    
    # Normaliser les scores
    for col in ['f_score', 'mi_score', 'rf_importance']:
        all_scores[col] = (all_scores[col] - all_scores[col].min()) / (all_scores[col].max() - all_scores[col].min())
    
    # Score combiné
    all_scores['combined_score'] = (all_scores['f_score'] + all_scores['mi_score'] + all_scores['rf_importance']) / 3
    all_scores = all_scores.sort_values('combined_score', ascending=False)
    
    # Sélectionner les top features
    selected_features = all_scores.head(k)['feature'].tolist()
    
    print(f"\n📊 Top 10 features sélectionnées:")
    for idx, row in all_scores.head(10).iterrows():
        print(f"   - {row['feature']}: {row['combined_score']:.4f}")
    
    # Vérifier si une feature domine
    top_score = all_scores.iloc[0]['combined_score']
    second_score = all_scores.iloc[1]['combined_score']
    if top_score > 0.8:  # Si une feature a un score trop élevé
        print(f"\n⚠️ ATTENTION: {all_scores.iloc[0]['feature']} domine avec un score de {top_score:.4f}")
        print("   Exclusion de cette feature pour éviter l'overfitting...")
        # Exclure complètement la feature dominante
        selected_features = all_scores.iloc[1:k+1]['feature'].tolist()
    else:
        selected_features = all_scores.head(k)['feature'].tolist()
    
    # Diversifier les features sélectionnées
    # Si certaines features sont trop corrélées, n'en garder qu'une partie
    final_features = []
    for i, feat in enumerate(selected_features):
        if i == 0 or not any(feat.startswith(f.split('_')[0]) for f in final_features[-3:]):
            final_features.append(feat)
        if len(final_features) >= k:
            break
    
    # Compléter si nécessaire
    if len(final_features) < k:
        remaining = [f for f in selected_features if f not in final_features]
        final_features.extend(remaining[:k-len(final_features)])
    
    return final_features[:k]

def create_safe_rolling_features(df, feature_cols, windows=[7, 30], min_periods_ratio=0.7):
    """Crée des rolling features sans data leakage"""
    df = df.copy()
    
    for col in feature_cols[:10]:  # Limiter aux 10 premières
        for window in windows:
            # S'assurer d'avoir assez de données
            min_periods = int(window * min_periods_ratio)
            
            # Rolling mean et std seulement
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(
                window=window, min_periods=min_periods
            ).mean()
            
            df[f'{col}_roll_std_{window}'] = df[col].rolling(
                window=window, min_periods=min_periods
            ).std()
            
            # Ratio avec la moyenne mobile
            df[f'{col}_roll_ratio_{window}'] = df[col] / (df[f'{col}_roll_mean_{window}'] + 1e-8)
    
    return df

def create_safe_lag_features(df, feature_cols, lags=[1, 7, 30]):
    """Crée des lag features simples"""
    df = df.copy()
    
    for col in feature_cols[:10]:  # Limiter aux 10 premières
        for lag in lags:
            # Simple lag
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            # Différence avec le lag
            df[f'{col}_diff_{lag}'] = df[col] - df[col].shift(lag)
    
    return df

def feature_engineering_controlled(train_df, test_df):
    """Feature engineering contrôlé pour éviter l'overfitting"""
    print("\n🔧 Feature Engineering Contrôlé...")
    
    # 1. Features temporelles
    print("  - Features temporelles...")
    train_df = create_date_features(train_df)
    test_df = create_date_features(test_df)
    
    # 2. Identifier les colonnes numériques
    feature_cols = [col for col in train_df.columns 
                   if col.startswith('Features_') and train_df[col].dtype in ['float64', 'int64']]
    
    print(f"  - {len(feature_cols)} features numériques trouvées")
    
    # 3. Sélection des features importantes AVANT de créer des dérivées
    selected_features = select_important_features(train_df, feature_cols, k=30)
    
    # 4. Créer des features dérivées SEULEMENT pour les features sélectionnées
    print("\n  - Création de rolling features...")
    train_df = create_safe_rolling_features(train_df, selected_features, windows=[7, 30])
    test_df = create_safe_rolling_features(test_df, selected_features, windows=[7, 30])
    
    print("  - Création de lag features...")
    train_df = create_safe_lag_features(train_df, selected_features, lags=[1, 7, 30])
    test_df = create_safe_lag_features(test_df, selected_features, lags=[1, 7, 30])
    
    # 5. Interactions simples (seulement top 5)
    print("  - Création d'interactions...")
    top_5_features = selected_features[:5]
    for i in range(len(top_5_features)):
        for j in range(i+1, len(top_5_features)):
            col1, col2 = top_5_features[i], top_5_features[j]
            train_df[f'{col1}_x_{col2}'] = train_df[col1] * train_df[col2]
            test_df[f'{col1}_x_{col2}'] = test_df[col1] * test_df[col2]
            
            # Ratio
            train_df[f'{col1}_div_{col2}'] = train_df[col1] / (train_df[col2] + 1e-8)
            test_df[f'{col1}_div_{col2}'] = test_df[col1] / (test_df[col2] + 1e-8)
    
    print(f"\n✅ Feature engineering terminé:")
    print(f"   - Train: {train_df.shape}")
    print(f"   - Test: {test_df.shape}")
    
    return train_df, test_df

# ========================================
# MODÉLISATION AVEC VALIDATION TEMPORELLE
# ========================================

class TimeSeriesModels:
    """Ensemble de modèles avec validation temporelle stricte"""
    
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.models = {}
        self.oof_predictions = None
        self.feature_importance = None
        
    def get_models(self):
        """Retourne les modèles optimisés pour time series"""
        return {
            'lgb': lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                n_estimators=1000,
                learning_rate=0.01,
                num_leaves=31,
                max_depth=5,
                min_child_samples=20,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1
            ),
            'xgb': xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=5,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbosity=0
            ),
            'cat': CatBoostRegressor(
                iterations=1000,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=3,
                min_data_in_leaf=20,
                random_strength=0.5,
                bagging_temperature=0.2,
                od_type='Iter',
                od_wait=50,
                random_state=42,
                verbose=False
            ),
            'rf': RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.01,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42
            )
        }
    
    def fit(self, X, y, feature_names=None):
        """Entraîne les modèles avec TimeSeriesSplit"""
        print("\n🏗️ Entraînement avec validation temporelle stricte...")
        
        # TimeSeriesSplit pour respecter l'ordre temporel
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # Initialiser les arrays pour stocker les résultats
        self.oof_predictions = {}
        all_scores = {}
        
        # Pour chaque modèle
        for name, model in self.get_models().items():
            print(f"\n  📊 Modèle: {name}")
            
            oof_pred = np.zeros(len(y))
            scores = []
            feature_importances = []
            
            # Cross-validation temporelle
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                print(f"    Fold {fold_idx + 1}/{self.n_splits}", end=' ')
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Entraînement
                if name == 'lgb':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                    )
                elif name == 'xgb':
                    # Pour XGBoost, utiliser les callbacks
                    model.set_params(early_stopping_rounds=50, eval_metric='rmse')
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                elif name == 'cat':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train)
                
                # Prédictions
                val_pred = model.predict(X_val)
                oof_pred[val_idx] = val_pred
                
                # Score
                score = np.sqrt(mean_squared_error(y_val, val_pred))
                scores.append(score)
                print(f"RMSE: {score:.6f}")
                
                # Feature importance (pour les modèles qui le supportent)
                if hasattr(model, 'feature_importances_'):
                    feature_importances.append(model.feature_importances_)
            
            # Stocker les résultats
            self.oof_predictions[name] = oof_pred
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            all_scores[name] = {'mean': mean_score, 'std': std_score}
            
            print(f"    Score moyen: {mean_score:.6f} (+/- {std_score:.6f})")
            
            # Réentraîner sur toutes les données
            print(f"    Entraînement sur l'ensemble complet...")
            # Créer un nouveau modèle sans early stopping pour l'entraînement final
            if name == 'lgb':
                final_model = lgb.LGBMRegressor(**self.get_models()['lgb'].get_params())
                final_model.fit(X, y, callbacks=[lgb.log_evaluation(0)])
                self.models[name] = final_model
            elif name == 'xgb':
                final_model = xgb.XGBRegressor(**self.get_models()['xgb'].get_params())
                final_model.set_params(early_stopping_rounds=None)
                final_model.fit(X, y, verbose=False)
                self.models[name] = final_model
            elif name == 'cat':
                final_model = CatBoostRegressor(**self.get_models()['cat'].get_params())
                final_model.fit(X, y, verbose=False)
                self.models[name] = final_model
            else:
                model.fit(X, y)
                self.models[name] = model
            
            # Moyenne des feature importances
            if feature_importances and feature_names is not None:
                mean_importance = np.mean(feature_importances, axis=0)
                self.feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    f'{name}_importance': mean_importance
                }).sort_values(f'{name}_importance', ascending=False)
        
        # Optimiser les poids de l'ensemble
        self._optimize_weights(y)
        
        return all_scores
    
    def _optimize_weights(self, y_true):
        """Optimise les poids de l'ensemble"""
        from scipy.optimize import minimize
        
        print("\n🎯 Optimisation des poids de l'ensemble...")
        
        # Convertir en array
        oof_array = np.column_stack([self.oof_predictions[name] for name in self.models.keys()])
        
        def objective(weights):
            weighted_pred = np.average(oof_array, axis=1, weights=weights)
            return np.sqrt(mean_squared_error(y_true, weighted_pred))
        
        # Contraintes
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(self.models))]
        
        # Point de départ
        x0 = np.ones(len(self.models)) / len(self.models)
        
        # Optimisation
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        self.weights = dict(zip(self.models.keys(), result.x))
        
        print("\n📊 Poids optimaux:")
        for name, weight in self.weights.items():
            print(f"   - {name}: {weight:.3f}")
        
        print(f"\n   Score ensemble (RMSE): {result.fun:.6f}")
    
    def predict(self, X):
        """Prédictions avec l'ensemble pondéré"""
        predictions = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # Moyenne pondérée
        pred_array = np.column_stack([predictions[name] for name in self.models.keys()])
        weights_array = np.array([self.weights[name] for name in self.models.keys()])
        
        return np.average(pred_array, axis=1, weights=weights_array)

# ========================================
# POST-PROCESSING
# ========================================

def post_process_predictions(predictions, train_target, test_size):
    """Post-processing intelligent des prédictions"""
    print("\n🎯 Post-processing des prédictions...")
    
    # 1. Gérer les valeurs négatives si nécessaire
    min_train = train_target.min()
    if min_train >= 0 and predictions.min() < 0:
        print(f"  - Correction des valeurs négatives ({np.sum(predictions < 0)} valeurs)")
        predictions = np.maximum(predictions, 0)
    
    # 2. Gérer les valeurs extrêmes
    # Utiliser des percentiles plus conservateurs
    lower_bound = np.percentile(train_target, 1)
    upper_bound = np.percentile(train_target, 99)
    
    n_clipped = np.sum((predictions < lower_bound) | (predictions > upper_bound))
    if n_clipped > 0:
        print(f"  - Clipping de {n_clipped} valeurs extrêmes")
        predictions = np.clip(predictions, lower_bound, upper_bound)
    
    # 3. Ajustement de la distribution si nécessaire
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    train_mean = train_target.mean()
    train_std = train_target.std()
    
    ratio_std = pred_std / train_std
    if ratio_std < 0.5 or ratio_std > 2.0:
        print(f"  - Ajustement de la variance (ratio: {ratio_std:.3f})")
        # Ajustement conservateur
        target_std = train_std * 0.9  # Légèrement moins de variance
        predictions = (predictions - pred_mean) * (target_std / pred_std) + train_mean
    
    # 4. Lissage temporel pour les séries temporelles
    # Appliquer un lissage léger pour réduire le bruit
    if test_size > 100:
        print("  - Lissage temporel appliqué")
        # Moving average avec fenêtre très petite
        window = 3
        predictions_smooth = pd.Series(predictions).rolling(
            window=window, center=True, min_periods=1
        ).mean().values
        # Blend: 80% original, 20% lissé
        predictions = 0.8 * predictions + 0.2 * predictions_smooth
    
    print(f"\n📊 Statistiques après post-processing:")
    print(f"   - Moyenne: {predictions.mean():.6f}")
    print(f"   - Std: {predictions.std():.6f}")
    print(f"   - Min: {predictions.min():.6f}")
    print(f"   - Max: {predictions.max():.6f}")
    
    return predictions

# ========================================
# PIPELINE PRINCIPAL
# ========================================

def main():
    """Pipeline principal optimisé"""
    
    # 1. Charger les données
    train_df, test_df = load_data()
    
    # 2. Feature Engineering contrôlé
    train_df, test_df = feature_engineering_controlled(train_df, test_df)
    
    # 3. Préparer les données
    exclude_cols = ['Dates', 'ToPredict', 'ID']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Gérer les valeurs manquantes
    print("\n🧹 Gestion des valeurs manquantes...")
    # Pour les lags et rolling: forward fill puis 0
    for col in feature_cols:
        if 'lag' in col or 'roll' in col or 'diff' in col:
            train_df[col] = train_df[col].fillna(method='ffill').fillna(0)
            test_df[col] = test_df[col].fillna(method='ffill').fillna(0)
        else:
            # Pour les autres: médiane
            median_val = train_df[col].median()
            train_df[col] = train_df[col].fillna(median_val)
            test_df[col] = test_df[col].fillna(median_val)
    
    # Vérification finale
    print(f"  - NaN dans train: {train_df[feature_cols].isna().sum().sum()}")
    print(f"  - NaN dans test: {test_df[feature_cols].isna().sum().sum()}")
    
    X_train = train_df[feature_cols].values
    y_train = train_df['ToPredict'].values
    X_test = test_df[feature_cols].values
    
    print(f"\n📐 Dimensions finales:")
    print(f"   - X_train: {X_train.shape}")
    print(f"   - X_test: {X_test.shape}")
    
    # 4. Normalisation robuste
    print("\n🔧 Normalisation des données...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Entraînement des modèles
    ts_models = TimeSeriesModels(n_splits=5)
    scores = ts_models.fit(X_train_scaled, y_train, feature_names=feature_cols)
    
    # 6. Prédictions
    print("\n📈 Génération des prédictions...")
    predictions = ts_models.predict(X_test_scaled)
    
    # 7. Post-processing
    predictions = post_process_predictions(predictions, y_train, len(X_test))
    
    # 8. Sauvegarder les résultats
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'ToPredict': predictions
    })
    submission.to_csv('submission_timeseries_optimized.csv', index=False)
    print("\n✅ Fichier de soumission créé: submission_timeseries_optimized.csv")
    
    # 9. Analyse des feature importances
    if ts_models.feature_importance is not None:
        print("\n🔍 Top 10 features importantes:")
        print(ts_models.feature_importance.head(10))
    
    # 10. Visualisation
    plt.figure(figsize=(15, 10))
    
    # Distribution comparison
    plt.subplot(2, 2, 1)
    plt.hist(predictions, bins=50, alpha=0.7, label='Predictions', density=True, color='blue')
    plt.hist(y_train, bins=50, alpha=0.7, label='Train', density=True, color='orange')
    plt.xlabel('Target Value')
    plt.ylabel('Density')
    plt.title('Distribution Comparison')
    plt.legend()
    
    # Predictions over time
    plt.subplot(2, 2, 2)
    plt.plot(test_df['Dates'], predictions, alpha=0.7, color='blue')
    plt.xlabel('Date')
    plt.ylabel('Target')
    plt.title('Predictions Over Time')
    plt.xticks(rotation=45)
    
    # QQ Plot
    plt.subplot(2, 2, 3)
    from scipy import stats
    stats.probplot(predictions, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Predictions')
    
    # Feature importance (if available)
    if ts_models.feature_importance is not None:
        plt.subplot(2, 2, 4)
        top_features = ts_models.feature_importance.head(15)
        plt.barh(range(len(top_features)), top_features.iloc[:, 1])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 15 Feature Importances')
        plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('predictions_timeseries_analysis.png', dpi=300, bbox_inches='tight')
    print("\n📊 Graphiques sauvegardés: predictions_timeseries_analysis.png")
    
    # Retourner les scores pour analyse
    return scores, predictions

if __name__ == "__main__":
    scores, predictions = main()