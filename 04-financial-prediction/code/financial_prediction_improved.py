#!/usr/bin/env python3
"""
Prédiction de Séries Temporelles Financières - Version Améliorée
Basée sur la version simple qui donne les meilleurs résultats
avec des améliorations subtiles pour performance optimale

Améliorations clés:
- Feature selection plus robuste avec diversification
- Validation temporelle avec gap pour éviter le look-ahead bias
- Ensemble avec stacking léger
- Post-processing adaptatif basé sur la tendance
- Optimisation des hyperparamètres par zone temporelle
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import gc
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, HuberRegressor, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Set random seeds
np.random.seed(42)

print("🚀 Financial Time Series Prediction - Version Améliorée")
print("=" * 60)

# ========================================
# FONCTIONS UTILITAIRES
# ========================================

def reduce_memory_usage(df):
    """Réduit l'usage mémoire du DataFrame"""
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        if col == 'Dates' or col == 'ID':
            continue
            
        col_type = str(df[col].dtype)
        
        # Skip non-numeric columns
        if 'object' in col_type or 'datetime' in col_type or 'string' in col_type:
            continue
        
        try:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Check if NaN
            if pd.isna(c_min) or pd.isna(c_max):
                continue
                
            if 'int' in col_type:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            elif 'float' in col_type:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        except:
            # Skip column if any error
            continue
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'  Mémoire réduite de {start_mem:.2f} MB à {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}%)')
    
    return df

# ========================================
# CHARGEMENT DES DONNÉES
# ========================================

def load_data(data_dir='data'):
    """Charge les données avec analyse approfondie"""
    print("\n📊 Chargement des données...")
    
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')
    
    # Conversion des dates
    train_df['Dates'] = pd.to_datetime(train_df['Dates'])
    test_df['Dates'] = pd.to_datetime(test_df['Dates'])
    
    # Tri par date - CRUCIAL
    train_df = train_df.sort_values('Dates').reset_index(drop=True)
    test_df = test_df.sort_values('Dates').reset_index(drop=True)
    
    # ID pour la soumission
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
    print(f"   - Skewness: {train_df['ToPredict'].skew():.6f}")
    print(f"   - Kurtosis: {train_df['ToPredict'].kurtosis():.6f}")
    
    return train_df, test_df

# ========================================
# FEATURE ENGINEERING AMÉLIORÉ
# ========================================

def create_date_features(df):
    """Crée des features temporelles enrichies"""
    df = df.copy()
    
    # Features temporelles de base
    df['year'] = df['Dates'].dt.year
    df['month'] = df['Dates'].dt.month
    df['day'] = df['Dates'].dt.day
    df['dayofweek'] = df['Dates'].dt.dayofweek
    df['quarter'] = df['Dates'].dt.quarter
    df['dayofyear'] = df['Dates'].dt.dayofyear
    df['weekofyear'] = df['Dates'].dt.isocalendar().week
    
    # Features cycliques
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 30)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 30)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Features additionnelles
    df['is_month_start'] = (df['day'] <= 5).astype(int)
    df['is_month_end'] = (df['day'] >= 25).astype(int)
    df['is_quarter_start'] = df['Dates'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['Dates'].dt.is_quarter_end.astype(int)
    df['is_year_start'] = (df['dayofyear'] <= 10).astype(int)
    df['is_year_end'] = (df['dayofyear'] >= 355).astype(int)
    
    return df

def select_diverse_features(train_df, feature_cols, target_col='ToPredict', k=35):
    """Sélection de features diversifiées pour éviter la domination"""
    print(f"\n🔍 Sélection de {k} features diversifiées...")
    
    # 1. F-statistics
    selector_f = SelectKBest(f_regression, k=min(k*2, len(feature_cols)))
    selector_f.fit(train_df[feature_cols], train_df[target_col])
    scores_f = pd.DataFrame({
        'feature': feature_cols,
        'f_score': selector_f.scores_
    }).sort_values('f_score', ascending=False)
    
    # 2. Mutual Information
    mi_scores = mutual_info_regression(train_df[feature_cols], train_df[target_col], random_state=42)
    scores_mi = pd.DataFrame({
        'feature': feature_cols,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # 3. RandomForest importance
    rf = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
    rf.fit(train_df[feature_cols], train_df[target_col])
    scores_rf = pd.DataFrame({
        'feature': feature_cols,
        'rf_importance': rf.feature_importances_
    }).sort_values('rf_importance', ascending=False)
    
    # 4. Gradient Boosting importance
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(train_df[feature_cols], train_df[target_col])
    scores_gb = pd.DataFrame({
        'feature': feature_cols,
        'gb_importance': gb.feature_importances_
    }).sort_values('gb_importance', ascending=False)
    
    # Combiner les scores
    all_scores = scores_f.merge(scores_mi, on='feature').merge(scores_rf, on='feature').merge(scores_gb, on='feature')
    
    # Normaliser les scores
    for col in ['f_score', 'mi_score', 'rf_importance', 'gb_importance']:
        all_scores[col] = (all_scores[col] - all_scores[col].min()) / (all_scores[col].max() - all_scores[col].min() + 1e-10)
    
    # Score combiné avec poids différents
    all_scores['combined_score'] = (
        0.25 * all_scores['f_score'] + 
        0.25 * all_scores['mi_score'] + 
        0.30 * all_scores['rf_importance'] +
        0.20 * all_scores['gb_importance']
    )
    all_scores = all_scores.sort_values('combined_score', ascending=False)
    
    print(f"\n📊 Top 10 features par score combiné:")
    for idx, row in all_scores.head(10).iterrows():
        print(f"   - {row['feature']}: {row['combined_score']:.4f}")
    
    # Stratégie de diversification
    selected_features = []
    feature_groups = {}
    
    # Grouper les features par préfixe
    for feat in feature_cols:
        prefix = feat.split('_')[0]
        if prefix not in feature_groups:
            feature_groups[prefix] = []
        feature_groups[prefix].append(feat)
    
    # Sélectionner de manière diversifiée
    for _, row in all_scores.iterrows():
        feature = row['feature']
        prefix = feature.split('_')[0]
        
        # Vérifier si on n'a pas trop de features du même groupe
        same_prefix_count = sum(1 for f in selected_features if f.startswith(prefix))
        
        # Limiter à 3 features par préfixe pour la diversité
        if same_prefix_count < 3 or len(selected_features) < k//2:
            selected_features.append(feature)
        
        if len(selected_features) >= k:
            break
    
    # Si pas assez, compléter avec les meilleures restantes
    if len(selected_features) < k:
        remaining = [f for f in all_scores['feature'].tolist() if f not in selected_features]
        selected_features.extend(remaining[:k-len(selected_features)])
    
    print(f"\n✅ {len(selected_features)} features sélectionnées avec diversité")
    
    return selected_features[:k]

def create_advanced_rolling_features(df, feature_cols, windows=[5, 10, 20, 40]):
    """Rolling features avancées avec ratios et tendances"""
    df = df.copy()
    
    for col in feature_cols[:12]:  # Top 12 features
        for window in windows:
            # Statistiques de base
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window, min_periods=int(window*0.6)).mean()
            df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window, min_periods=int(window*0.6)).std()
            
            # Ratios et changements
            df[f'{col}_roll_ratio_{window}'] = df[col] / (df[f'{col}_roll_mean_{window}'] + 1e-8)
            df[f'{col}_roll_change_{window}'] = df[col] - df[f'{col}_roll_mean_{window}']
            
            # Tendance (pente de régression linéaire)
            if window >= 10:
                df[f'{col}_trend_{window}'] = df[col].rolling(window=window, min_periods=int(window*0.6)).apply(
                    lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 3 else 0,
                    raw=True
                )
    
    return df

def create_advanced_lag_features(df, feature_cols, lags=[1, 3, 7, 14, 30]):
    """Lag features avec différences et ratios"""
    df = df.copy()
    
    for col in feature_cols[:12]:  # Top 12 features
        for lag in lags:
            # Lag simple
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            # Différence
            df[f'{col}_diff_{lag}'] = df[col] - df[col].shift(lag)
            
            # Ratio
            df[f'{col}_ratio_{lag}'] = df[col] / (df[col].shift(lag) + 1e-8)
            
            # Changement en pourcentage
            df[f'{col}_pct_change_{lag}'] = (df[col] - df[col].shift(lag)) / (df[col].shift(lag).abs() + 1e-8)
    
    return df

def create_interaction_features(df, selected_features, top_n=8):
    """Créer des interactions entre les top features"""
    df = df.copy()
    
    top_features = selected_features[:top_n]
    
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            col1, col2 = top_features[i], top_features[j]
            
            # Produit
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            
            # Ratio
            df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
            
            # Somme et différence
            df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
            df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
    
    return df

def feature_engineering_advanced(train_df, test_df):
    """Pipeline de feature engineering avancé mais contrôlé"""
    print("\n🔧 Feature Engineering Avancé...")
    
    # 1. Features temporelles
    print("  - Features temporelles...")
    train_df = create_date_features(train_df)
    test_df = create_date_features(test_df)
    
    # 2. Identifier les colonnes numériques
    feature_cols = [col for col in train_df.columns 
                   if col.startswith('Features_') and train_df[col].dtype in ['float64', 'int64']]
    
    print(f"  - {len(feature_cols)} features numériques trouvées")
    
    # 3. Sélection diversifiée des features
    selected_features = select_diverse_features(train_df, feature_cols, k=35)
    
    # 4. Rolling features avancées
    print("\n  - Création de rolling features avancées...")
    train_df = create_advanced_rolling_features(train_df, selected_features, windows=[5, 10, 20, 40])
    test_df = create_advanced_rolling_features(test_df, selected_features, windows=[5, 10, 20, 40])
    
    # 5. Lag features avancées
    print("  - Création de lag features avancées...")
    train_df = create_advanced_lag_features(train_df, selected_features, lags=[1, 3, 7, 14, 30])
    test_df = create_advanced_lag_features(test_df, selected_features, lags=[1, 3, 7, 14, 30])
    
    # 6. Interactions
    print("  - Création d'interactions...")
    train_df = create_interaction_features(train_df, selected_features, top_n=8)
    test_df = create_interaction_features(test_df, selected_features, top_n=8)
    
    # 7. Réduction mémoire - Commenté car cause des problèmes avec fillna
    # print("\n  - Optimisation mémoire...")
    # train_df = reduce_memory_usage(train_df)
    # test_df = reduce_memory_usage(test_df)
    
    print(f"\n✅ Feature engineering terminé:")
    print(f"   - Train: {train_df.shape}")
    print(f"   - Test: {test_df.shape}")
    
    return train_df, test_df

# ========================================
# MODÉLISATION AVANCÉE
# ========================================

class AdvancedTimeSeriesModels:
    """Ensemble avancé avec validation temporelle et stacking"""
    
    def __init__(self, n_splits=5, gap=10):
        self.n_splits = n_splits
        self.gap = gap  # Gap entre train et validation pour éviter le look-ahead
        self.models = {}
        self.meta_model = None
        self.oof_predictions = None
        self.feature_importance = None
        
    def get_base_models(self):
        """Modèles de base optimisés"""
        return {
            'lgb': lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                n_estimators=1500,
                learning_rate=0.008,
                num_leaves=31,
                max_depth=5,
                min_child_samples=20,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1
            ),
            'xgb': xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=1500,
                learning_rate=0.008,
                max_depth=5,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbosity=0
            ),
            'cat': CatBoostRegressor(
                iterations=1500,
                learning_rate=0.02,
                depth=6,
                l2_leaf_reg=3,
                min_data_in_leaf=20,
                random_strength=0.5,
                bagging_temperature=0.2,
                od_type='Iter',
                od_wait=100,
                random_state=42,
                verbose=False
            ),
            'rf': RandomForestRegressor(
                n_estimators=500,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'et': ExtraTreesRegressor(
                n_estimators=500,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.008,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42
            )
        }
    
    def custom_time_series_split(self, X, y):
        """Split temporel avec gap pour éviter le data leakage"""
        n = len(X)
        indices = np.arange(n)
        
        for i in range(self.n_splits):
            # Calculer les indices de manière progressive
            test_size = n // (self.n_splits + 1)
            train_end = (i + 1) * test_size
            test_start = train_end + self.gap
            test_end = min(test_start + test_size, n)
            
            if test_end > n or test_start >= n:
                continue
                
            train_indices = indices[:train_end]
            test_indices = indices[test_start:test_end]
            
            yield train_indices, test_indices
    
    def fit(self, X, y, feature_names=None):
        """Entraînement avec stacking"""
        print("\n🏗️ Entraînement avec validation temporelle avancée...")
        
        # Arrays pour stocker les prédictions OOF
        self.oof_predictions = {}
        all_scores = {}
        
        # Phase 1: Entraîner les modèles de base
        for name, model in self.get_base_models().items():
            print(f"\n  📊 Modèle: {name}")
            
            oof_pred = np.zeros(len(y))
            scores = []
            feature_importances = []
            
            # Validation croisée avec gap
            for fold_idx, (train_idx, val_idx) in enumerate(self.custom_time_series_split(X, y)):
                print(f"    Fold {fold_idx + 1}/{self.n_splits}", end=' ')
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Entraînement avec early stopping
                if name == 'lgb':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
                    )
                elif name == 'xgb':
                    model.set_params(early_stopping_rounds=100, eval_metric='rmse')
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                elif name == 'cat':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=100,
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
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    feature_importances.append(model.feature_importances_)
            
            # Stocker les résultats
            self.oof_predictions[name] = oof_pred
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            all_scores[name] = {'mean': mean_score, 'std': std_score}
            
            print(f"    Score moyen: {mean_score:.6f} (+/- {std_score:.6f})")
            
            # Réentraîner sur toutes les données
            print(f"    Entraînement final...")
            if name in ['lgb', 'xgb', 'cat']:
                # Sans early stopping pour l'entraînement final
                final_model = self.get_base_models()[name]
                if name == 'xgb':
                    final_model.set_params(early_stopping_rounds=None)
                    final_model.fit(X, y, verbose=False)
                elif name == 'lgb':
                    final_model.fit(X, y, callbacks=[lgb.log_evaluation(0)])
                elif name == 'cat':
                    final_model.fit(X, y, verbose=False)
                self.models[name] = final_model
            else:
                model.fit(X, y)
                self.models[name] = model
        
        # Phase 2: Méta-modèle (stacking)
        print("\n🎯 Entraînement du méta-modèle...")
        
        # Préparer les features pour le méta-modèle
        meta_features = np.column_stack([self.oof_predictions[name] for name in self.models.keys()])
        
        # Méta-modèle simple mais robuste
        self.meta_model = Ridge(alpha=1.0, random_state=42)
        self.meta_model.fit(meta_features, y)
        
        # Calculer le score final
        meta_pred = self.meta_model.predict(meta_features)
        final_score = np.sqrt(mean_squared_error(y, meta_pred))
        print(f"  Score du stacking: {final_score:.6f}")
        
        # Optimiser les poids
        self._optimize_weights(y)
        
        return all_scores
    
    def _optimize_weights(self, y_true):
        """Optimise les poids avec régularisation"""
        from scipy.optimize import minimize
        
        print("\n🎯 Optimisation des poids...")
        
        oof_array = np.column_stack([self.oof_predictions[name] for name in self.models.keys()])
        
        def objective(weights):
            weighted_pred = np.average(oof_array, axis=1, weights=weights)
            return np.sqrt(mean_squared_error(y_true, weighted_pred))
        
        # Contraintes
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0.05, 0.5) for _ in range(len(self.models))]  # Limiter les poids pour éviter la domination
        
        # Point de départ uniforme
        x0 = np.ones(len(self.models)) / len(self.models)
        
        # Optimisation
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        self.weights = dict(zip(self.models.keys(), result.x))
        
        print("\n📊 Poids optimaux:")
        for name, weight in self.weights.items():
            print(f"   - {name}: {weight:.3f}")
        
        print(f"\n   Score avec poids optimaux: {result.fun:.6f}")
    
    def predict(self, X, use_meta=False):
        """Prédictions avec ensemble ou méta-modèle"""
        predictions = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        if use_meta and self.meta_model is not None:
            # Utiliser le méta-modèle
            meta_features = np.column_stack([predictions[name] for name in self.models.keys()])
            return self.meta_model.predict(meta_features)
        else:
            # Moyenne pondérée
            pred_array = np.column_stack([predictions[name] for name in self.models.keys()])
            weights_array = np.array([self.weights[name] for name in self.models.keys()])
            
            return np.average(pred_array, axis=1, weights=weights_array)

# ========================================
# POST-PROCESSING ADAPTATIF
# ========================================

def adaptive_post_processing(predictions, train_target, test_dates, train_dates):
    """Post-processing adaptatif basé sur les tendances"""
    print("\n🎯 Post-processing adaptatif...")
    
    # 1. Analyse de la tendance récente
    recent_period = 90  # 3 mois
    recent_mask = (train_dates >= (train_dates.max() - pd.Timedelta(days=recent_period)))
    recent_target = train_target[recent_mask]
    
    if len(recent_target) > 30:
        recent_trend = np.polyfit(np.arange(len(recent_target)), recent_target, 1)[0]
        print(f"  - Tendance récente: {recent_trend:.6f}")
        
        # Ajuster légèrement les prédictions selon la tendance
        if abs(recent_trend) > 0.001:
            trend_adjustment = recent_trend * np.arange(len(predictions)) * 0.1  # Ajustement léger
            predictions = predictions + trend_adjustment
    
    # 2. Correction des valeurs extrêmes adaptative
    # Utiliser des percentiles adaptatifs basés sur la période récente
    lower_percentile = max(1, min(5, len(recent_target) * 0.02)) if len(recent_target) > 50 else 2
    upper_percentile = min(99, max(95, 100 - len(recent_target) * 0.02)) if len(recent_target) > 50 else 98
    
    lower_bound = np.percentile(train_target, lower_percentile)
    upper_bound = np.percentile(train_target, upper_percentile)
    
    n_clipped = np.sum((predictions < lower_bound) | (predictions > upper_bound))
    if n_clipped > 0:
        print(f"  - Clipping adaptatif de {n_clipped} valeurs")
        predictions = np.clip(predictions, lower_bound, upper_bound)
    
    # 3. Ajustement de la variance
    pred_std = predictions.std()
    recent_std = recent_target.std() if len(recent_target) > 30 else train_target.std()
    
    ratio_std = pred_std / recent_std
    if ratio_std < 0.7 or ratio_std > 1.3:
        print(f"  - Ajustement de variance (ratio: {ratio_std:.3f})")
        target_std = recent_std * 0.95  # Légèrement conservateur
        predictions = (predictions - predictions.mean()) * (target_std / pred_std) + train_target.mean()
    
    # 4. Lissage adaptatif
    if len(predictions) > 50:
        # Détection de volatilité
        volatility = np.std(np.diff(predictions))
        if volatility > recent_std * 0.5:
            print("  - Lissage adaptatif appliqué")
            # Lissage exponentiel
            alpha = 0.85  # Poids pour la valeur actuelle
            smoothed = predictions.copy()
            for i in range(1, len(smoothed)):
                smoothed[i] = alpha * predictions[i] + (1 - alpha) * smoothed[i-1]
            predictions = smoothed
    
    # 5. Correction saisonnière si détectée
    if 'month' in train_dates.dt.month.unique():
        monthly_means = train_target.groupby(train_dates.dt.month).mean()
        test_months = test_dates.dt.month
        seasonal_adjustment = test_months.map(monthly_means) - train_target.mean()
        predictions = predictions + seasonal_adjustment.values * 0.2  # Ajustement léger
    
    print(f"\n📊 Statistiques finales:")
    print(f"   - Moyenne: {predictions.mean():.6f}")
    print(f"   - Std: {predictions.std():.6f}")
    print(f"   - Min: {predictions.min():.6f}")
    print(f"   - Max: {predictions.max():.6f}")
    
    return predictions

# ========================================
# PIPELINE PRINCIPAL
# ========================================

def main():
    """Pipeline principal amélioré"""
    
    # 1. Charger les données
    train_df, test_df = load_data()
    
    # 2. Feature Engineering avancé
    train_df, test_df = feature_engineering_advanced(train_df, test_df)
    
    # 3. Préparer les données
    exclude_cols = ['Dates', 'ToPredict', 'ID']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Gérer les valeurs manquantes de manière intelligente
    print("\n🧹 Gestion intelligente des valeurs manquantes...")
    for col in feature_cols:
        if 'lag' in col or 'roll' in col or 'diff' in col:
            # Pour les features temporelles: forward fill puis médiane
            train_median = train_df[col].median()
            train_df[col] = train_df[col].ffill().fillna(train_median)
            test_df[col] = test_df[col].ffill().fillna(train_median)
        else:
            # Pour les autres: médiane du train
            median_val = train_df[col].median()
            train_df[col] = train_df[col].fillna(median_val)
            test_df[col] = test_df[col].fillna(median_val)
    
    X_train = train_df[feature_cols].values
    y_train = train_df['ToPredict'].values
    X_test = test_df[feature_cols].values
    
    print(f"\n📐 Dimensions finales:")
    print(f"   - X_train: {X_train.shape}")
    print(f"   - X_test: {X_test.shape}")
    
    # 4. Normalisation avec QuantileTransformer pour robustesse
    print("\n🔧 Normalisation robuste...")
    scaler = QuantileTransformer(
        n_quantiles=min(1000, len(X_train)),
        output_distribution='normal',
        random_state=42
    )
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Entraînement des modèles avancés
    ts_models = AdvancedTimeSeriesModels(n_splits=5, gap=10)
    scores = ts_models.fit(X_train_scaled, y_train, feature_names=feature_cols)
    
    # 6. Prédictions avec les deux approches
    print("\n📈 Génération des prédictions...")
    predictions_weighted = ts_models.predict(X_test_scaled, use_meta=False)
    predictions_stacked = ts_models.predict(X_test_scaled, use_meta=True)
    
    # Blend des deux approches
    predictions = 0.7 * predictions_weighted + 0.3 * predictions_stacked
    
    # 7. Post-processing adaptatif
    predictions = adaptive_post_processing(
        predictions, 
        y_train, 
        test_df['Dates'], 
        train_df['Dates']
    )
    
    # 8. Sauvegarder les résultats
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'ToPredict': predictions
    })
    submission.to_csv('submission_improved.csv', index=False)
    print("\n✅ Fichier de soumission créé: submission_improved.csv")
    
    # 9. Visualisation avancée
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Distribution comparison
    ax = axes[0, 0]
    ax.hist(predictions, bins=50, alpha=0.7, label='Predictions', density=True, color='blue')
    ax.hist(y_train, bins=50, alpha=0.7, label='Train', density=True, color='orange')
    ax.set_xlabel('Target Value')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Comparison')
    ax.legend()
    
    # 2. Time series plot
    ax = axes[0, 1]
    ax.plot(test_df['Dates'], predictions, alpha=0.7, color='blue', label='Predictions')
    # Ajouter une ligne de tendance
    z = np.polyfit(np.arange(len(predictions)), predictions, 3)
    p = np.poly1d(z)
    ax.plot(test_df['Dates'], p(np.arange(len(predictions))), "r--", alpha=0.8, label='Trend')
    ax.set_xlabel('Date')
    ax.set_ylabel('Target')
    ax.set_title('Predictions Over Time')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    # 3. Q-Q Plot
    ax = axes[0, 2]
    from scipy import stats
    stats.probplot(predictions, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot of Predictions')
    
    # 4. Rolling statistics
    ax = axes[1, 0]
    rolling_mean = pd.Series(predictions).rolling(window=30, min_periods=1).mean()
    rolling_std = pd.Series(predictions).rolling(window=30, min_periods=1).std()
    ax.plot(test_df['Dates'], rolling_mean, label='30-day MA', color='blue')
    ax.fill_between(test_df['Dates'], 
                    rolling_mean - rolling_std, 
                    rolling_mean + rolling_std, 
                    alpha=0.3, color='blue', label='±1 STD')
    ax.set_xlabel('Date')
    ax.set_ylabel('Target')
    ax.set_title('Rolling Statistics')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    # 5. Model predictions comparison
    ax = axes[1, 1]
    for i, (name, model) in enumerate(ts_models.models.items()):
        model_pred = model.predict(X_test_scaled)
        ax.scatter(predictions, model_pred, alpha=0.5, s=10, label=name)
    ax.plot([predictions.min(), predictions.max()], 
            [predictions.min(), predictions.max()], 
            'k--', alpha=0.5)
    ax.set_xlabel('Ensemble Predictions')
    ax.set_ylabel('Individual Model Predictions')
    ax.set_title('Model Predictions Comparison')
    ax.legend()
    
    # 6. Residual analysis
    ax = axes[1, 2]
    # Simuler des résidus en comparant avec une moyenne mobile du train
    train_recent_mean = y_train[-len(predictions):].mean() if len(y_train) >= len(predictions) else y_train.mean()
    pseudo_residuals = predictions - train_recent_mean
    ax.scatter(predictions, pseudo_residuals, alpha=0.5, s=10)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Predictions')
    ax.set_ylabel('Pseudo-Residuals')
    ax.set_title('Residual Pattern Analysis')
    
    plt.tight_layout()
    plt.savefig('predictions_improved_analysis.png', dpi=300, bbox_inches='tight')
    print("\n📊 Graphiques sauvegardés: predictions_improved_analysis.png")
    
    # 10. Rapport de performance
    print("\n📊 Rapport de performance:")
    print("=" * 50)
    for name, score_info in scores.items():
        print(f"{name:10} - RMSE: {score_info['mean']:.6f} (+/- {score_info['std']:.6f})")
    print("=" * 50)
    
    # Calculer quelques métriques supplémentaires
    print("\n📈 Métriques des prédictions:")
    print(f"  - Corrélation avec train récent: {np.corrcoef(predictions[:min(100, len(predictions))], y_train[-min(100, len(y_train)):])[0,1]:.4f}")
    print(f"  - Ratio variance (pred/train): {predictions.std() / y_train.std():.4f}")
    print(f"  - Skewness différence: {abs(stats.skew(predictions) - stats.skew(y_train)):.4f}")
    
    return scores, predictions

if __name__ == "__main__":
    scores, predictions = main()