#!/usr/bin/env python3
"""
Prédiction de Séries Temporelles Financières - Version Enhanced
Améliorations subtiles pour optimiser encore les performances

Nouvelles améliorations:
- Features de tendance et saisonnalité
- Meta-learner avec stacking
- Optimisation Bayésienne légère
- Blend de différentes normalisations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from typing import List, Tuple, Dict
import gc
from tqdm import tqdm
import joblib

warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, HuberRegressor, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Optimisation
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Set random seeds
np.random.seed(42)

print("🚀 Financial Time Series Prediction - Version Enhanced")
print("=" * 60)

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
    
    # Tri par date
    train_df = train_df.sort_values('Dates').reset_index(drop=True)
    test_df = test_df.sort_values('Dates').reset_index(drop=True)
    
    # Utiliser les dates comme ID pour le test
    if 'ID' not in test_df.columns:
        test_df['ID'] = test_df['Dates'].dt.strftime('%Y-%m-%d')
    
    print(f"✅ Données chargées:")
    print(f"   - Train: {train_df.shape}")
    print(f"   - Test: {test_df.shape}")
    
    return train_df, test_df

# ========================================
# FEATURE ENGINEERING AVANCÉ
# ========================================

def create_trend_features(df, feature_cols, window=30):
    """Crée des features de tendance"""
    df = df.copy()
    
    for col in feature_cols[:5]:  # Top 5 features seulement
        # Tendance linéaire sur fenêtre glissante
        df[f'{col}_trend_{window}'] = df[col].rolling(window).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else np.nan,
            raw=True
        )
        
        # Accélération (dérivée seconde)
        df[f'{col}_acceleration_{window}'] = df[f'{col}_trend_{window}'].diff()
    
    return df

def create_seasonality_features(df, feature_cols):
    """Détecte et encode la saisonnalité"""
    df = df.copy()
    
    # Saisonnalité hebdomadaire et mensuelle
    for col in feature_cols[:5]:
        # Moyenne par jour de la semaine
        dow_mean = df.groupby(df['Dates'].dt.dayofweek)[col].transform('mean')
        df[f'{col}_dow_seasonal'] = df[col] - dow_mean
        
        # Moyenne par jour du mois
        dom_mean = df.groupby(df['Dates'].dt.day)[col].transform('mean')
        df[f'{col}_dom_seasonal'] = df[col] - dom_mean
    
    return df

def create_advanced_interactions(df, feature_cols):
    """Crée des interactions plus sophistiquées"""
    df = df.copy()
    
    # Top 3 features seulement pour limiter l'explosion
    top_features = feature_cols[:3]
    
    for i, col1 in enumerate(top_features):
        for j, col2 in enumerate(top_features[i+1:], i+1):
            # Interactions non-linéaires
            df[f'{col1}_sqrt_x_{col2}'] = np.sqrt(np.abs(df[col1])) * df[col2]
            df[f'{col1}_log1p_ratio_{col2}'] = np.log1p(np.abs(df[col1])) / (np.abs(df[col2]) + 1e-8)
    
    return df

def feature_engineering_enhanced(train_df, test_df):
    """Feature engineering amélioré"""
    print("\n🔧 Feature Engineering Enhanced...")
    
    # 1. Features temporelles de base
    print("  - Features temporelles...")
    train_df = create_date_features(train_df)
    test_df = create_date_features(test_df)
    
    # 2. Sélection intelligente des features
    feature_cols = [col for col in train_df.columns 
                   if col.startswith('Features_') and train_df[col].dtype in ['float64', 'int64']]
    
    selected_features = select_smart_features(train_df, feature_cols, k=25)
    
    # 3. Features de tendance
    print("  - Features de tendance...")
    train_df = create_trend_features(train_df, selected_features, window=30)
    test_df = create_trend_features(test_df, selected_features, window=30)
    
    # 4. Features de saisonnalité
    print("  - Features de saisonnalité...")
    train_df = create_seasonality_features(train_df, selected_features)
    test_df = create_seasonality_features(test_df, selected_features)
    
    # 5. Rolling features optimisées
    print("  - Rolling features...")
    train_df = create_optimized_rolling_features(train_df, selected_features)
    test_df = create_optimized_rolling_features(test_df, selected_features)
    
    # 6. Lag features avec decay
    print("  - Lag features avec decay...")
    train_df = create_weighted_lag_features(train_df, selected_features)
    test_df = create_weighted_lag_features(test_df, selected_features)
    
    # 7. Interactions avancées
    print("  - Interactions avancées...")
    train_df = create_advanced_interactions(train_df, selected_features)
    test_df = create_advanced_interactions(test_df, selected_features)
    
    print(f"\n✅ Feature engineering terminé: {len(train_df.columns)} features")
    
    return train_df, test_df

def create_date_features(df):
    """Features temporelles avec encodage amélioré"""
    df = df.copy()
    
    # Features de base
    df['year'] = df['Dates'].dt.year
    df['month'] = df['Dates'].dt.month
    df['day'] = df['Dates'].dt.day
    df['dayofweek'] = df['Dates'].dt.dayofweek
    df['quarter'] = df['Dates'].dt.quarter
    df['dayofyear'] = df['Dates'].dt.dayofyear
    df['weekofyear'] = df['Dates'].dt.isocalendar().week
    
    # Encodage cyclique amélioré
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 30)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 30)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Features spéciales
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = (df['day'] <= 7).astype(int)
    df['is_month_end'] = (df['day'] >= 24).astype(int)
    df['is_quarter_start'] = ((df['month'] - 1) % 3 == 0) & (df['day'] <= 7)
    df['is_quarter_end'] = ((df['month']) % 3 == 0) & (df['day'] >= 24)
    
    return df

def select_smart_features(train_df, feature_cols, k=25):
    """Sélection de features améliorée avec diversité"""
    print(f"\n🔍 Sélection intelligente de {k} features...")
    
    # Méthode 1: F-statistics
    selector_f = SelectKBest(f_regression, k=min(len(feature_cols), 50))
    selector_f.fit(train_df[feature_cols], train_df['ToPredict'])
    scores_f = pd.DataFrame({
        'feature': feature_cols,
        'f_score': selector_f.scores_
    })
    
    # Méthode 2: Mutual Information
    mi_scores = mutual_info_regression(
        train_df[feature_cols], 
        train_df['ToPredict'], 
        n_neighbors=5,
        random_state=42
    )
    scores_mi = pd.DataFrame({
        'feature': feature_cols,
        'mi_score': mi_scores
    })
    
    # Méthode 3: Random Forest avec plus d'arbres
    rf = RandomForestRegressor(
        n_estimators=200, 
        max_depth=8, 
        min_samples_leaf=20,
        random_state=42, 
        n_jobs=-1
    )
    rf.fit(train_df[feature_cols], train_df['ToPredict'])
    scores_rf = pd.DataFrame({
        'feature': feature_cols,
        'rf_importance': rf.feature_importances_
    })
    
    # Méthode 4: Gradient Boosting pour capturer les interactions
    gb = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    gb.fit(train_df[feature_cols], train_df['ToPredict'])
    scores_gb = pd.DataFrame({
        'feature': feature_cols,
        'gb_importance': gb.feature_importances_
    })
    
    # Combiner tous les scores
    all_scores = scores_f.merge(scores_mi, on='feature')\
                        .merge(scores_rf, on='feature')\
                        .merge(scores_gb, on='feature')
    
    # Normaliser chaque score
    for col in ['f_score', 'mi_score', 'rf_importance', 'gb_importance']:
        min_val = all_scores[col].min()
        max_val = all_scores[col].max()
        all_scores[col] = (all_scores[col] - min_val) / (max_val - min_val + 1e-8)
    
    # Score pondéré avec plus de poids sur RF et GB
    all_scores['combined_score'] = (
        0.15 * all_scores['f_score'] + 
        0.15 * all_scores['mi_score'] + 
        0.35 * all_scores['rf_importance'] +
        0.35 * all_scores['gb_importance']
    )
    
    all_scores = all_scores.sort_values('combined_score', ascending=False)
    
    # Exclure les features trop dominantes
    if all_scores.iloc[0]['combined_score'] > 0.8:
        print(f"⚠️ Exclusion de {all_scores.iloc[0]['feature']} (score: {all_scores.iloc[0]['combined_score']:.3f})")
        all_scores = all_scores.iloc[1:]
    
    # Sélection avec diversité
    selected = []
    feature_groups = {}
    
    for _, row in all_scores.iterrows():
        feature = row['feature']
        # Extraire le numéro de base de la feature
        base_num = int(feature.split('_')[1])
        group = base_num // 10  # Grouper par dizaines
        
        # Limiter à 3 features par groupe pour la diversité
        if group not in feature_groups:
            feature_groups[group] = 0
        
        if feature_groups[group] < 3:
            selected.append(feature)
            feature_groups[group] += 1
            
        if len(selected) >= k:
            break
    
    # Si pas assez, compléter avec les meilleures restantes
    if len(selected) < k:
        remaining = [f for f in all_scores['feature'] if f not in selected]
        selected.extend(remaining[:k-len(selected)])
    
    print(f"✅ Features sélectionnées avec diversité:")
    print(f"   Top 5: {selected[:5]}")
    
    return selected[:k]

def create_optimized_rolling_features(df, feature_cols):
    """Rolling features avec fenêtres adaptatives"""
    df = df.copy()
    
    # Fenêtres multiples pour capturer différents horizons
    windows = [7, 14, 30, 60]
    
    for col in feature_cols[:8]:  # Top 8 features
        for window in windows:
            min_periods = max(window // 3, 3)
            
            # Statistiques robustes
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(
                window=window, min_periods=min_periods
            ).mean()
            
            df[f'{col}_roll_std_{window}'] = df[col].rolling(
                window=window, min_periods=min_periods
            ).std()
            
            # Quantiles pour capturer la distribution
            df[f'{col}_roll_q25_{window}'] = df[col].rolling(
                window=window, min_periods=min_periods
            ).quantile(0.25)
            
            df[f'{col}_roll_q75_{window}'] = df[col].rolling(
                window=window, min_periods=min_periods
            ).quantile(0.75)
            
            # Range normalisé
            roll_max = df[col].rolling(window=window, min_periods=min_periods).max()
            roll_min = df[col].rolling(window=window, min_periods=min_periods).min()
            df[f'{col}_roll_range_norm_{window}'] = (df[col] - roll_min) / (roll_max - roll_min + 1e-8)
    
    return df

def create_weighted_lag_features(df, feature_cols):
    """Lag features avec decay exponentiel"""
    df = df.copy()
    
    lags = [1, 3, 7, 14, 30]
    decay_factor = 0.95  # Facteur de décroissance
    
    for col in feature_cols[:8]:
        # Lag simple
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Lag pondéré avec decay
        weighted_sum = pd.Series(0, index=df.index)
        weight_sum = 0
        
        for lag in range(1, 31):  # 30 jours de lag
            weight = decay_factor ** lag
            weighted_sum += df[col].shift(lag).fillna(0) * weight
            weight_sum += weight
        
        df[f'{col}_weighted_lag_30'] = weighted_sum / weight_sum
        
        # Différence avec lag pondéré
        df[f'{col}_diff_weighted'] = df[col] - df[f'{col}_weighted_lag_30']
    
    return df

# ========================================
# MODÉLISATION AVANCÉE
# ========================================

class EnhancedTimeSeriesModels:
    """Ensemble amélioré avec meta-learning"""
    
    def __init__(self, n_splits=5, use_optuna=False):
        self.n_splits = n_splits
        self.use_optuna = use_optuna
        self.models = {}
        self.meta_model = None
        self.oof_predictions = {}
        self.best_params = {}
        
    def get_base_models(self):
        """Modèles de base optimisés"""
        return {
            'lgb': lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                n_estimators=1000,
                learning_rate=0.01,
                num_leaves=31,
                max_depth=6,
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
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=6,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.7,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbosity=0
            ),
            'cat': CatBoostRegressor(
                iterations=1000,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=5,
                min_data_in_leaf=20,
                random_strength=0.5,
                bagging_temperature=0.3,
                border_count=128,
                od_type='Iter',
                od_wait=50,
                random_state=42,
                verbose=False
            ),
            'et': ExtraTreesRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ),
            'ridge': Ridge(
                alpha=10.0,
                random_state=42
            ),
            'nn': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.01,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
        }
    
    def optimize_hyperparameters(self, X_sample, y_sample, model_name):
        """Optimisation Bayésienne légère des hyperparamètres"""
        if not self.use_optuna:
            return self.get_base_models()[model_name]
        
        print(f"    Optimisation Optuna pour {model_name}...")
        
        def objective(trial):
            if model_name == 'lgb':
                params = {
                    'n_estimators': 1000,
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 50),
                    'max_depth': trial.suggest_int('max_depth', 4, 8),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 30),
                    'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.5),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 0.5),
                    'random_state': 42,
                    'verbose': -1
                }
                model = lgb.LGBMRegressor(**params)
            else:
                return 0  # Skip optimization for other models
            
            # Quick CV
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            for train_idx, val_idx in tscv.split(X_sample):
                X_train, X_val = X_sample[train_idx], X_sample[val_idx]
                y_train, y_val = y_sample[train_idx], y_sample[val_idx]
                
                if model_name == 'lgb':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
                    )
                else:
                    model.fit(X_train, y_train)
                
                pred = model.predict(X_val)
                scores.append(np.sqrt(mean_squared_error(y_val, pred)))
            
            return np.mean(scores)
        
        # Optimisation rapide
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        
        # Retourner le meilleur modèle
        if model_name == 'lgb':
            best_params = study.best_params
            best_params.update({
                'n_estimators': 1000,
                'random_state': 42,
                'verbose': -1
            })
            return lgb.LGBMRegressor(**best_params)
        
        return self.get_base_models()[model_name]
    
    def fit(self, X, y, feature_names=None):
        """Entraînement avec stacking amélioré"""
        print("\n🏗️ Entraînement Enhanced avec Meta-Learning...")
        
        # TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # Sample pour Optuna (si activé)
        if self.use_optuna:
            sample_size = min(1000, len(X))
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[sample_idx]
            y_sample = y[sample_idx]
        
        # Pour chaque modèle de base
        all_scores = {}
        
        for name in self.get_base_models().keys():
            print(f"\n  📊 Modèle: {name}")
            
            # Optimiser ou utiliser le modèle de base
            if self.use_optuna and name in ['lgb']:
                model = self.optimize_hyperparameters(X_sample, y_sample, name)
            else:
                model = self.get_base_models()[name]
            
            oof_pred = np.zeros(len(y))
            scores = []
            
            # Cross-validation
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                print(f"    Fold {fold_idx + 1}/{self.n_splits}", end=' ')
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Entraînement selon le type de modèle
                if name == 'lgb':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                    )
                elif name == 'xgb':
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
                
                score = np.sqrt(mean_squared_error(y_val, val_pred))
                scores.append(score)
                print(f"RMSE: {score:.6f}")
            
            # Stocker les résultats
            self.oof_predictions[name] = oof_pred
            mean_score = np.mean(scores)
            all_scores[name] = mean_score
            print(f"    Score moyen: {mean_score:.6f}")
            
            # Réentraîner sur toutes les données
            print(f"    Entraînement final...")
            if name == 'lgb':
                final_model = self.get_base_models()[name]
                final_model.fit(X, y, callbacks=[lgb.log_evaluation(0)])
            elif name == 'xgb':
                final_model = self.get_base_models()[name]
                final_model.set_params(early_stopping_rounds=None)
                final_model.fit(X, y, verbose=False)
            elif name == 'cat':
                final_model = self.get_base_models()[name]
                final_model.fit(X, y, verbose=False)
            else:
                final_model = self.get_base_models()[name]
                final_model.fit(X, y)
            
            self.models[name] = final_model
        
        # Meta-learning avec Ridge
        self._train_meta_learner(y)
        
        return all_scores
    
    def _train_meta_learner(self, y_true):
        """Entraîne un meta-learner pour combiner les prédictions"""
        print("\n🎯 Entraînement du Meta-Learner...")
        
        # Préparer les features pour le meta-model
        meta_features = np.column_stack([
            self.oof_predictions[name] for name in self.models.keys()
        ])
        
        # Ajouter des features dérivées
        meta_features_enhanced = np.hstack([
            meta_features,
            np.mean(meta_features, axis=1).reshape(-1, 1),
            np.std(meta_features, axis=1).reshape(-1, 1),
            np.max(meta_features, axis=1).reshape(-1, 1),
            np.min(meta_features, axis=1).reshape(-1, 1)
        ])
        
        # Entraîner plusieurs meta-models et sélectionner le meilleur
        meta_models = {
            'ridge': Ridge(alpha=1.0, random_state=42),
            'lasso': Lasso(alpha=0.01, random_state=42),
            'huber': HuberRegressor(epsilon=1.35, alpha=1.0)
        }
        
        best_score = float('inf')
        best_meta = None
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        for name, meta_model in meta_models.items():
            scores = []
            for train_idx, val_idx in tscv.split(meta_features_enhanced):
                X_train = meta_features_enhanced[train_idx]
                y_train = y_true[train_idx]
                X_val = meta_features_enhanced[val_idx]
                y_val = y_true[val_idx]
                
                meta_model.fit(X_train, y_train)
                pred = meta_model.predict(X_val)
                scores.append(np.sqrt(mean_squared_error(y_val, pred)))
            
            mean_score = np.mean(scores)
            if mean_score < best_score:
                best_score = mean_score
                best_meta = name
        
        print(f"   Meilleur meta-model: {best_meta} (RMSE: {best_score:.6f})")
        
        # Entraîner le meilleur meta-model sur toutes les données
        self.meta_model = meta_models[best_meta]
        self.meta_model.fit(meta_features_enhanced, y_true)
    
    def predict(self, X):
        """Prédictions avec meta-learning"""
        # Prédictions de base
        base_predictions = np.column_stack([
            self.models[name].predict(X) for name in self.models.keys()
        ])
        
        # Features pour le meta-model
        meta_features = np.hstack([
            base_predictions,
            np.mean(base_predictions, axis=1).reshape(-1, 1),
            np.std(base_predictions, axis=1).reshape(-1, 1),
            np.max(base_predictions, axis=1).reshape(-1, 1),
            np.min(base_predictions, axis=1).reshape(-1, 1)
        ])
        
        # Prédiction finale
        return self.meta_model.predict(meta_features)

# ========================================
# MULTI-SCALE PREDICTIONS
# ========================================

def create_multi_scale_predictions(models, X_train, y_train, X_test, feature_names):
    """Crée des prédictions à différentes échelles temporelles"""
    print("\n🔮 Génération de prédictions multi-échelles...")
    
    predictions = {}
    
    # 1. Modèle sur données complètes
    print("  - Échelle complète...")
    model_full = EnhancedTimeSeriesModels(n_splits=5)
    model_full.fit(X_train, y_train, feature_names)
    predictions['full'] = model_full.predict(X_test)
    
    # 2. Modèle sur données récentes (derniers 20%)
    print("  - Échelle récente...")
    recent_size = int(0.2 * len(X_train))
    X_recent = X_train[-recent_size:]
    y_recent = y_train[-recent_size:]
    
    model_recent = EnhancedTimeSeriesModels(n_splits=3)
    model_recent.fit(X_recent, y_recent, feature_names)
    predictions['recent'] = model_recent.predict(X_test)
    
    # 3. Blend des échelles
    # Plus de poids sur le modèle complet, mais tenir compte du modèle récent
    final_predictions = 0.7 * predictions['full'] + 0.3 * predictions['recent']
    
    return final_predictions, predictions

# ========================================
# POST-PROCESSING AVANCÉ
# ========================================

def advanced_post_processing(predictions, train_target, test_size):
    """Post-processing avancé avec plusieurs techniques"""
    print("\n🎯 Post-processing avancé...")
    
    # 1. Correction des valeurs négatives si nécessaire
    if train_target.min() >= 0:
        predictions = np.maximum(predictions, 0)
    
    # 2. Clipping intelligent basé sur la distribution du train
    # Utiliser une approche plus souple
    lower_percentile = 0.5
    upper_percentile = 99.5
    
    lower_bound = np.percentile(train_target, lower_percentile)
    upper_bound = np.percentile(train_target, upper_percentile)
    
    # Permettre un peu de dépassement
    margin = 0.1 * (upper_bound - lower_bound)
    predictions = np.clip(predictions, lower_bound - margin, upper_bound + margin)
    
    # 3. Ajustement de la distribution
    # Matching des moments
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    train_mean = train_target.mean()
    train_std = train_target.std()
    
    # Ajustement conservateur
    if pred_std > 0:
        # Ajuster variance graduellement
        target_std = train_std * 0.95  # Légèrement moins de variance
        adjustment_factor = np.clip(target_std / pred_std, 0.8, 1.2)
        predictions = (predictions - pred_mean) * adjustment_factor + train_mean
    
    # 4. Lissage adaptatif
    if test_size > 50:
        # Appliquer un lissage plus fort sur les séries volatiles
        volatility = pd.Series(predictions).rolling(10, center=True, min_periods=5).std()
        high_volatility_mask = volatility > volatility.median()
        
        # Lissage différencié
        smooth_light = pd.Series(predictions).rolling(3, center=True, min_periods=1).mean()
        smooth_heavy = pd.Series(predictions).rolling(7, center=True, min_periods=3).mean()
        
        predictions = np.where(
            high_volatility_mask,
            0.5 * predictions + 0.5 * smooth_heavy,
            0.8 * predictions + 0.2 * smooth_light
        )
    
    # 5. Correction des outliers
    # Identifier et corriger les valeurs trop extrêmes
    z_scores = np.abs((predictions - np.mean(predictions)) / np.std(predictions))
    outlier_mask = z_scores > 3
    
    if outlier_mask.any():
        # Remplacer par la médiane locale
        median_window = pd.Series(predictions).rolling(21, center=True, min_periods=11).median()
        predictions[outlier_mask] = median_window[outlier_mask]
    
    print(f"✅ Post-processing terminé:")
    print(f"   - Moyenne: {predictions.mean():.6f}")
    print(f"   - Std: {predictions.std():.6f}")
    print(f"   - Plage: [{predictions.min():.6f}, {predictions.max():.6f}]")
    
    return predictions

# ========================================
# PIPELINE PRINCIPAL
# ========================================

def main():
    """Pipeline principal enhanced"""
    
    # 1. Charger les données
    train_df, test_df = load_data()
    
    # 2. Feature Engineering avancé
    train_df, test_df = feature_engineering_enhanced(train_df, test_df)
    
    # 3. Préparer les données
    exclude_cols = ['Dates', 'ToPredict', 'ID']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Gérer les valeurs manquantes de manière sophistiquée
    print("\n🧹 Gestion avancée des valeurs manquantes...")
    
    # Pour les features temporelles: interpolation
    time_features = [col for col in feature_cols if any(x in col for x in ['trend', 'seasonal', 'roll', 'lag'])]
    for col in time_features:
        train_df[col] = train_df[col].interpolate(method='linear', limit_direction='both')
        test_df[col] = test_df[col].interpolate(method='linear', limit_direction='both')
    
    # Pour les autres: médiane
    other_features = [col for col in feature_cols if col not in time_features]
    for col in other_features:
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col] = test_df[col].fillna(median_val)
    
    # Vérifier qu'il n'y a plus de NaN
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    test_df[feature_cols] = test_df[feature_cols].fillna(0)
    
    X_train = train_df[feature_cols].values
    y_train = train_df['ToPredict'].values
    X_test = test_df[feature_cols].values
    
    print(f"\n📐 Dimensions finales:")
    print(f"   - X_train: {X_train.shape}")
    print(f"   - X_test: {X_test.shape}")
    
    # 4. Multi-normalisation et blend
    print("\n🔧 Multi-normalisation...")
    
    # Trois types de normalisation
    scalers = {
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'quantile': QuantileTransformer(n_quantiles=100, random_state=42)
    }
    
    predictions_by_scaler = {}
    
    for scaler_name, scaler in scalers.items():
        print(f"\n  Normalisation {scaler_name}...")
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Modèle pour cette normalisation
        model = EnhancedTimeSeriesModels(n_splits=5, use_optuna=False)
        model.fit(X_train_scaled, y_train, feature_names=feature_cols)
        
        predictions_by_scaler[scaler_name] = model.predict(X_test_scaled)
    
    # 5. Blend des différentes normalisations
    print("\n🎨 Blending des prédictions...")
    # Pondération basée sur la performance historique
    weights = {
        'standard': 0.4,
        'robust': 0.4,
        'quantile': 0.2
    }
    
    predictions = sum(
        weights[name] * preds 
        for name, preds in predictions_by_scaler.items()
    )
    
    # 6. Post-processing avancé
    predictions = advanced_post_processing(predictions, y_train, len(X_test))
    
    # 7. Sauvegarder les résultats
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'ToPredict': predictions
    })
    submission.to_csv('submission_enhanced.csv', index=False)
    print("\n✅ Fichier de soumission créé: submission_enhanced.csv")
    
    # 8. Analyse finale
    print("\n📊 Analyse finale des prédictions:")
    print(f"   Distribution train - mean: {y_train.mean():.6f}, std: {y_train.std():.6f}")
    print(f"   Distribution pred  - mean: {predictions.mean():.6f}, std: {predictions.std():.6f}")
    print(f"   Ratio std (pred/train): {predictions.std()/y_train.std():.3f}")
    
    # 9. Visualisation améliorée
    create_enhanced_visualizations(train_df, test_df, y_train, predictions)
    
    return predictions

def create_enhanced_visualizations(train_df, test_df, y_train, predictions):
    """Crée des visualisations améliorées"""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Série temporelle complète
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(train_df['Dates'], y_train, label='Train', alpha=0.7, linewidth=1)
    ax1.plot(test_df['Dates'], predictions, label='Predictions', alpha=0.8, linewidth=1.5)
    ax1.set_title('Série Temporelle Complète')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Target')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Distribution comparison
    ax2 = plt.subplot(3, 2, 2)
    ax2.hist(y_train, bins=50, alpha=0.6, label='Train', density=True)
    ax2.hist(predictions, bins=50, alpha=0.6, label='Predictions', density=True)
    ax2.set_title('Comparaison des Distributions')
    ax2.set_xlabel('Valeur')
    ax2.set_ylabel('Densité')
    ax2.legend()
    
    # 3. Rolling statistics
    ax3 = plt.subplot(3, 2, 3)
    train_rolling_mean = pd.Series(y_train).rolling(30).mean()
    pred_rolling_mean = pd.Series(predictions).rolling(30).mean()
    ax3.plot(train_df['Dates'], train_rolling_mean, label='Train MA30', alpha=0.8)
    ax3.plot(test_df['Dates'], pred_rolling_mean, label='Pred MA30', alpha=0.8)
    ax3.set_title('Moyennes Mobiles 30 jours')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('MA30')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. QQ Plot
    ax4 = plt.subplot(3, 2, 4)
    from scipy import stats
    stats.probplot(predictions, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot des Prédictions')
    
    # 5. Volatility over time
    ax5 = plt.subplot(3, 2, 5)
    train_vol = pd.Series(y_train).rolling(30).std()
    pred_vol = pd.Series(predictions).rolling(30).std()
    ax5.plot(train_df['Dates'], train_vol, label='Train Volatility', alpha=0.8)
    ax5.plot(test_df['Dates'], pred_vol, label='Pred Volatility', alpha=0.8)
    ax5.set_title('Volatilité (Std Mobile 30j)')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Volatilité')
    ax5.legend()
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. Monthly aggregation
    ax6 = plt.subplot(3, 2, 6)
    test_df['predictions'] = predictions
    monthly_pred = test_df.groupby(test_df['Dates'].dt.to_period('M'))['predictions'].mean()
    train_df['ToPredict_temp'] = y_train
    monthly_train = train_df.groupby(train_df['Dates'].dt.to_period('M'))['ToPredict_temp'].mean()
    
    ax6.plot(monthly_train.index.astype(str), monthly_train.values, label='Train Monthly', marker='o')
    ax6.plot(monthly_pred.index.astype(str), monthly_pred.values, label='Pred Monthly', marker='s')
    ax6.set_title('Moyennes Mensuelles')
    ax6.set_xlabel('Mois')
    ax6.set_ylabel('Moyenne')
    ax6.legend()
    ax6.tick_params(axis='x', rotation=90)
    
    # Nettoyer les colonnes temporaires
    train_df.drop('ToPredict_temp', axis=1, inplace=True, errors='ignore')
    test_df.drop('predictions', axis=1, inplace=True, errors='ignore')
    
    plt.tight_layout()
    plt.savefig('predictions_enhanced_analysis.png', dpi=300, bbox_inches='tight')
    print("\n📊 Graphiques sauvegardés: predictions_enhanced_analysis.png")
    plt.close()

if __name__ == "__main__":
    predictions = main()