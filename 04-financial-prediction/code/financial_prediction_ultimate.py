#!/usr/bin/env python3
"""
Prédiction de Séries Temporelles Financières - Version Ultimate Performance
Version avancée avec toutes les optimisations pour atteindre le meilleur score

Améliorations par rapport à la version précédente :
- Feature engineering encore plus poussé
- Stacking avancé avec méta-modèle
- Optimisation Bayésienne des hyperparamètres
- Target engineering et post-processing
- Blending de plusieurs approches
"""

# ========================================
# 1. IMPORTS ET CONFIGURATION
# ========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
from typing import List, Tuple, Dict, Optional
import gc
import joblib
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import TimeSeriesSplit, KFold, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Optimisation
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Set random seeds
np.random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("🚀 Financial Time Series Prediction - Ultimate Performance Version")
print("=" * 60)

# ========================================
# 2. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ========================================

def load_data(data_dir='data'):
    """Charge les données d'entraînement et de test"""
    print("\n📊 Chargement des données...")
    
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')
    
    # Conversion des dates
    train_df['Dates'] = pd.to_datetime(train_df['Dates'])
    test_df['Dates'] = pd.to_datetime(test_df['Dates'])
    
    # Tri par date
    train_df = train_df.sort_values('Dates').reset_index(drop=True)
    test_df = test_df.sort_values('Dates').reset_index(drop=True)
    
    print(f"✅ Données chargées:")
    print(f"   - Train: {train_df.shape}")
    print(f"   - Test: {test_df.shape}")
    
    return train_df, test_df

# ========================================
# 3. FEATURE ENGINEERING AVANCÉ
# ========================================

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée les features temporelles avec encodage cyclique"""
    # Features temporelles de base
    df['year'] = df['Dates'].dt.year
    df['month'] = df['Dates'].dt.month
    df['day'] = df['Dates'].dt.day
    df['day_of_week'] = df['Dates'].dt.dayofweek
    df['day_of_year'] = df['Dates'].dt.dayofyear
    df['quarter'] = df['Dates'].dt.quarter
    df['week_of_year'] = df['Dates'].dt.isocalendar().week
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df['Dates'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Dates'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['Dates'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['Dates'].dt.is_quarter_end.astype(int)
    
    # Encodage cyclique
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    return df

def create_rolling_features_advanced(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Crée des features de rolling window avancées"""
    windows = [3, 5, 7, 10, 15, 20, 30, 50, 100]
    top_features = feature_cols[:15]  # Plus de features
    
    for window in windows:
        for col in top_features:
            # Statistiques de base
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
            df[f'{col}_roll_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
            df[f'{col}_roll_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
            
            # Statistiques avancées
            df[f'{col}_roll_skew_{window}'] = df[col].rolling(window=window, min_periods=1).skew()
            df[f'{col}_roll_kurt_{window}'] = df[col].rolling(window=window, min_periods=1).kurt()
            df[f'{col}_roll_median_{window}'] = df[col].rolling(window=window, min_periods=1).median()
            
            # Quantiles
            df[f'{col}_roll_q25_{window}'] = df[col].rolling(window=window, min_periods=1).quantile(0.25)
            df[f'{col}_roll_q75_{window}'] = df[col].rolling(window=window, min_periods=1).quantile(0.75)
            
            # Range et ratios
            df[f'{col}_roll_range_{window}'] = df[f'{col}_roll_max_{window}'] - df[f'{col}_roll_min_{window}']
            df[f'{col}_roll_mean_std_ratio_{window}'] = df[f'{col}_roll_mean_{window}'] / (df[f'{col}_roll_std_{window}'] + 1e-8)
    
    return df

def create_lag_features_advanced(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Crée des lag features avancées"""
    lags = [1, 2, 3, 5, 7, 10, 15, 20, 30]
    top_features = feature_cols[:10]
    
    for lag in lags:
        for col in top_features:
            # Lag simple
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            # Différences
            df[f'{col}_diff_{lag}'] = df[col] - df[col].shift(lag)
            
            # Ratios
            df[f'{col}_ratio_{lag}'] = df[col] / (df[col].shift(lag) + 1e-8)
            
            # Pourcentage de changement
            df[f'{col}_pct_change_{lag}'] = df[col].pct_change(lag)
    
    return df

def create_technical_indicators_advanced(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Crée des indicateurs techniques avancés"""
    top_features = feature_cols[:10]
    
    for col in top_features:
        # RSI multiple periods
        for period in [7, 14, 21, 28]:
            df[f'{col}_rsi_{period}'] = calculate_rsi(df[col], window=period)
        
        # Bollinger Bands multiple parameters
        for window, num_std in [(10, 2), (20, 2), (20, 3)]:
            bb = calculate_bollinger_bands(df[col], window=window, num_std=num_std)
            df[f'{col}_bb_upper_{window}_{num_std}'] = bb['upper']
            df[f'{col}_bb_lower_{window}_{num_std}'] = bb['lower']
            df[f'{col}_bb_width_{window}_{num_std}'] = bb['width']
            df[f'{col}_bb_position_{window}_{num_std}'] = bb['position']
        
        # MACD
        ema_12 = df[col].ewm(span=12, adjust=False).mean()
        ema_26 = df[col].ewm(span=26, adjust=False).mean()
        df[f'{col}_macd'] = ema_12 - ema_26
        df[f'{col}_macd_signal'] = df[f'{col}_macd'].ewm(span=9, adjust=False).mean()
        df[f'{col}_macd_diff'] = df[f'{col}_macd'] - df[f'{col}_macd_signal']
        
        # Stochastic Oscillator
        for period in [14, 21]:
            low_min = df[col].rolling(window=period).min()
            high_max = df[col].rolling(window=period).max()
            df[f'{col}_stoch_{period}'] = 100 * (df[col] - low_min) / (high_max - low_min + 1e-8)
    
    return df

def create_interaction_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Crée des features d'interaction"""
    top_features = feature_cols[:8]
    
    # Interactions multiplicatives
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            df[f'mult_{top_features[i]}_{top_features[j]}'] = df[top_features[i]] * df[top_features[j]]
    
    # Ratios importants
    important_ratios = [
        ('Features_34', 'Features_35'),  # Volatility ratios
        ('Features_79', 'Features_85'),  # Market indicators
        ('Features_93', 'Features_36'),  # Risk measures
    ]
    
    for num, denom in important_ratios:
        if num in df.columns and denom in df.columns:
            df[f'ratio_{num}_{denom}'] = df[num] / (df[denom] + 1e-8)
    
    return df

def create_advanced_transformations(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Applique des transformations avancées"""
    top_features = feature_cols[:20]
    
    for col in top_features:
        # Transformations non-linéaires
        df[f'{col}_log1p'] = np.log1p(np.abs(df[col]))
        df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
        df[f'{col}_square'] = df[col] ** 2
        df[f'{col}_cube'] = df[col] ** 3
        
        # Transformations trigonométriques
        df[f'{col}_sin'] = np.sin(df[col])
        df[f'{col}_cos'] = np.cos(df[col])
        
        # Binning
        df[f'{col}_bin'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
    
    return df

def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calcule le Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
    """Calcule les Bollinger Bands"""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    band_width = upper_band - lower_band
    band_position = (series - lower_band) / (band_width + 1e-10)
    
    return {
        'upper': upper_band,
        'lower': lower_band,
        'width': band_width,
        'position': band_position
    }

def engineer_features_ultimate(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Pipeline complet de feature engineering avancé"""
    print(f"\n🔧 Feature Engineering Ultimate {'(Train)' if is_train else '(Test)'}...")
    
    df_feat = df.copy()
    feature_cols = [col for col in df.columns if col.startswith('Features_')]
    
    # 1. Features temporelles
    print("  - Features temporelles...")
    df_feat = create_temporal_features(df_feat)
    
    # 2. Rolling features avancées
    print("  - Rolling features avancées...")
    df_feat = create_rolling_features_advanced(df_feat, feature_cols)
    
    # 3. Lag features avancées
    print("  - Lag features avancées...")
    df_feat = create_lag_features_advanced(df_feat, feature_cols)
    
    # 4. Indicateurs techniques avancés
    print("  - Indicateurs techniques avancés...")
    df_feat = create_technical_indicators_advanced(df_feat, feature_cols)
    
    # 5. Features d'interaction
    print("  - Features d'interaction...")
    df_feat = create_interaction_features(df_feat, feature_cols)
    
    # 6. Transformations avancées
    print("  - Transformations avancées...")
    df_feat = create_advanced_transformations(df_feat, feature_cols)
    
    # 7. Traitement des valeurs manquantes
    df_feat = df_feat.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # 8. Supprimer les colonnes avec variance nulle
    if is_train:
        # Exclure la colonne Dates et autres colonnes non numériques
        numeric_cols = df_feat.select_dtypes(include=[np.number]).columns
        zero_var_cols = []
        
        for col in numeric_cols:
            if df_feat[col].var() == 0:
                zero_var_cols.append(col)
        
        df_feat = df_feat.drop(columns=zero_var_cols)
        # Sauvegarder pour appliquer au test
        joblib.dump(zero_var_cols, 'zero_var_cols.pkl')
    else:
        # Charger et appliquer
        if os.path.exists('zero_var_cols.pkl'):
            zero_var_cols = joblib.load('zero_var_cols.pkl')
            df_feat = df_feat.drop(columns=zero_var_cols, errors='ignore')
    
    print(f"✅ Features créées: {df_feat.shape[1]} colonnes")
    gc.collect()
    
    return df_feat

# ========================================
# 4. SÉLECTION DE FEATURES
# ========================================

def select_features(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, 
                   n_features: int = 500) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sélection de features basée sur l'importance"""
    print(f"\n🎯 Sélection des {n_features} meilleures features...")
    
    # RandomForest pour l'importance
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Obtenir l'importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Sélectionner les top features
    selected_features = feature_importance.head(n_features)['feature'].tolist()
    
    print(f"  Top 10 features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"    - {row['feature']}: {row['importance']:.4f}")
    
    return X_train[selected_features], X_test[selected_features]

# ========================================
# 5. OPTIMISATION BAYÉSIENNE
# ========================================

def optimize_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, 
                  n_trials: int = 50) -> Dict:
    """Optimise les hyperparamètres avec Optuna"""
    print(f"\n🔍 Optimisation Bayésienne pour {model_name} ({n_trials} essais)...")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    def objective(trial):
        if model_name == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 10, 50),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7]),
                'random_state': 42,
                'n_jobs': -1
            }
            model = RandomForestRegressor(**params)
            
        elif model_name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42,
                'verbosity': -1,
                'n_jobs': -1
            }
            model = lgb.LGBMRegressor(**params)
            
        elif model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42,
                'n_jobs': -1
            }
            model = xgb.XGBRegressor(**params)
        
        # Cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=tscv, 
                               scoring='neg_mean_squared_error', n_jobs=1)
        return -scores.mean()
    
    # Créer l'étude
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
    
    # Optimiser
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"  Meilleur score: {study.best_value:.6f}")
    print(f"  Meilleurs paramètres: {study.best_params}")
    
    return study.best_params

# ========================================
# 6. MODÈLES ET STACKING
# ========================================

class AdvancedStacking:
    """Stacking avancé avec méta-modèle optimisé"""
    
    def __init__(self, base_models: List[Tuple[str, object]], meta_model: object,
                 use_probas: bool = False, cv_splits: int = 5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.use_probas = use_probas
        self.cv_splits = cv_splits
        self.oof_predictions = None
        self.trained_models = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Entraîne le stacking avec out-of-fold predictions"""
        print("\n🏗️ Entraînement du Stacking Avancé...")
        
        n_samples = len(X)
        n_models = len(self.base_models)
        
        # Initialiser les prédictions OOF
        self.oof_predictions = np.zeros((n_samples, n_models))
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        
        # Pour chaque modèle de base
        for i, (name, model) in enumerate(self.base_models):
            print(f"\n  Modèle {i+1}/{n_models}: {name}")
            model_oof = np.zeros(n_samples)
            
            # Cross-validation
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                
                # Cloner le modèle
                model_clone = model.__class__(**model.get_params())
                
                # Entraîner
                if isinstance(model_clone, lgb.LGBMRegressor):
                    model_clone.fit(
                        X_train_fold, y_train_fold,
                        eval_set=[(X_val_fold, y.iloc[val_idx])],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                    )
                elif hasattr(model_clone, 'fit') and 'eval_set' in model_clone.fit.__code__.co_varnames:
                    model_clone.fit(
                        X_train_fold, y_train_fold,
                        eval_set=[(X_val_fold, y.iloc[val_idx])],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                else:
                    model_clone.fit(X_train_fold, y_train_fold)
                
                # Prédire
                model_oof[val_idx] = model_clone.predict(X_val_fold)
            
            # Sauvegarder les prédictions OOF
            self.oof_predictions[:, i] = model_oof
            
            # Score OOF
            oof_score = np.sqrt(mean_squared_error(y, model_oof))
            print(f"    Score OOF (RMSE): {oof_score:.6f}")
        
        # Entraîner le méta-modèle
        print("\n  Entraînement du méta-modèle...")
        self.meta_model.fit(self.oof_predictions, y)
        
        # Réentraîner les modèles de base sur toutes les données
        print("\n  Réentraînement sur l'ensemble complet...")
        self.trained_models = []
        
        for name, model in self.base_models:
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X, y)
            self.trained_models.append((name, model_clone))
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Fait des prédictions"""
        n_models = len(self.trained_models)
        base_predictions = np.zeros((len(X), n_models))
        
        # Prédictions des modèles de base
        for i, (name, model) in enumerate(self.trained_models):
            base_predictions[:, i] = model.predict(X)
        
        # Prédiction finale avec le méta-modèle
        return self.meta_model.predict(base_predictions)

def get_optimized_models(X_train: pd.DataFrame, y_train: pd.Series, 
                        optimize: bool = True) -> Dict:
    """Retourne les modèles avec hyperparamètres optimisés"""
    
    if optimize:
        # Optimiser les principaux modèles
        rf_params = optimize_model('RandomForest', X_train, y_train, n_trials=30)
        lgb_params = optimize_model('LightGBM', X_train, y_train, n_trials=30)
        xgb_params = optimize_model('XGBoost', X_train, y_train, n_trials=30)
    else:
        # Paramètres par défaut optimisés
        rf_params = {
            'n_estimators': 300,
            'max_depth': 12,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        lgb_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'num_leaves': 50,
            'max_depth': 8,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1
        }
        xgb_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 8,
            'min_child_weight': 3,
            'gamma': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }
    
    models = {
        'RandomForest': RandomForestRegressor(**rf_params),
        'LightGBM': lgb.LGBMRegressor(**lgb_params),
        'XGBoost': xgb.XGBRegressor(**xgb_params),
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        ),
        'CatBoost': CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            random_state=42,
            verbose=False,
            thread_count=-1
        )
    }
    
    return models

# ========================================
# 7. POST-PROCESSING ET TARGET ENGINEERING
# ========================================

def target_transform(y: pd.Series, method: str = 'log') -> Tuple[pd.Series, object]:
    """Transforme la target pour améliorer la distribution"""
    if method == 'log':
        y_transformed = np.log1p(y)
        transformer = lambda x: np.log1p(x)
        inverse_transformer = lambda x: np.expm1(x)
    elif method == 'sqrt':
        y_transformed = np.sqrt(y)
        transformer = lambda x: np.sqrt(x)
        inverse_transformer = lambda x: x ** 2
    elif method == 'quantile':
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        y_transformed = pd.Series(qt.fit_transform(y.values.reshape(-1, 1)).ravel())
        transformer = qt
        inverse_transformer = lambda x: qt.inverse_transform(x.reshape(-1, 1)).ravel()
    else:
        y_transformed = y
        transformer = lambda x: x
        inverse_transformer = lambda x: x
    
    return y_transformed, inverse_transformer

def post_process_predictions(predictions: np.ndarray, train_target: pd.Series) -> np.ndarray:
    """Post-traitement des prédictions"""
    # Clipper aux valeurs min/max du train
    predictions = np.clip(predictions, 
                         train_target.min() * 0.9, 
                         train_target.max() * 1.1)
    
    # Smooth outliers
    q1 = np.percentile(predictions, 25)
    q3 = np.percentile(predictions, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Identifier et ajuster les outliers
    outlier_mask = (predictions < lower_bound) | (predictions > upper_bound)
    if outlier_mask.any():
        predictions[outlier_mask] = np.clip(predictions[outlier_mask], 
                                          lower_bound, upper_bound)
    
    return predictions

# ========================================
# 8. FONCTION PRINCIPALE
# ========================================

def main():
    """Fonction principale d'exécution"""
    
    # Configuration
    OPTIMIZE_HYPERPARAMS = False  # Mettre True pour optimisation complète
    N_FEATURES = 500  # Nombre de features à sélectionner
    USE_TARGET_TRANSFORM = True  # Transformer la target
    
    # 1. Chargement des données
    train_df, test_df = load_data()
    
    # 2. Feature Engineering Ultimate
    train_enhanced = engineer_features_ultimate(train_df, is_train=True)
    test_enhanced = engineer_features_ultimate(test_df, is_train=False)
    
    # 3. Préparation des données
    feature_cols = [col for col in train_enhanced.columns 
                    if col not in ['Dates', 'ToPredict']]
    
    X_train = train_enhanced[feature_cols]
    y_train = train_enhanced['ToPredict']
    X_test = test_enhanced[feature_cols]
    
    # 4. Sélection de features
    X_train_selected, X_test_selected = select_features(X_train, y_train, X_test, 
                                                       n_features=N_FEATURES)
    
    print(f"\n📐 Dimensions après sélection:")
    print(f"  - X_train: {X_train_selected.shape}")
    print(f"  - X_test: {X_test_selected.shape}")
    
    # 5. Target transformation
    if USE_TARGET_TRANSFORM:
        print("\n🔄 Transformation de la target...")
        y_train_transformed, inverse_transformer = target_transform(y_train, method='log')
    else:
        y_train_transformed = y_train
        inverse_transformer = lambda x: x
    
    # 6. Normalisation
    print("\n🔧 Normalisation des données...")
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_selected),
        columns=X_train_selected.columns,
        index=X_train_selected.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_selected),
        columns=X_test_selected.columns,
        index=X_test_selected.index
    )
    
    # 7. Obtenir les modèles optimisés
    models = get_optimized_models(X_train_scaled, y_train_transformed, 
                                 optimize=OPTIMIZE_HYPERPARAMS)
    
    # 8. Créer le stacking
    base_models = list(models.items())
    
    # Méta-modèle
    meta_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    # Entraîner le stacking
    stacking = AdvancedStacking(base_models, meta_model, cv_splits=5)
    stacking.fit(X_train_scaled, y_train_transformed)
    
    # 9. Prédictions
    print("\n📈 Génération des prédictions...")
    predictions_transformed = stacking.predict(X_test_scaled)
    
    # Inverse transform
    predictions = inverse_transformer(predictions_transformed)
    
    # Post-processing
    predictions = post_process_predictions(predictions, y_train)
    
    # 10. Blending avec différentes approches
    print("\n🎨 Blending final...")
    
    # Approche 1: Prédictions du stacking
    blend_predictions = predictions.copy()
    
    # Approche 2: Moyenne des top 3 modèles
    top_3_models = ['RandomForest', 'LightGBM', 'XGBoost']
    top_3_preds = []
    
    for name in top_3_models:
        model = models[name]
        model.fit(X_train_scaled, y_train_transformed)
        pred = inverse_transformer(model.predict(X_test_scaled))
        top_3_preds.append(pred)
    
    avg_top_3 = np.mean(top_3_preds, axis=0)
    
    # Blend final (80% stacking, 20% moyenne top 3)
    final_predictions = 0.8 * blend_predictions + 0.2 * avg_top_3
    
    # 11. Créer le fichier de soumission
    submission = pd.DataFrame({
        'ID': test_df['Dates'].dt.strftime('%Y-%m-%d'),
        'ToPredict': final_predictions
    })
    
    submission.to_csv('submission_ultimate.csv', index=False)
    print("\n✅ Fichier de soumission créé: submission_ultimate.csv")
    
    # 12. Statistiques finales
    print("\n📊 Statistiques des prédictions finales:")
    print(f"  - Moyenne: {final_predictions.mean():.6f}")
    print(f"  - Std: {final_predictions.std():.6f}")
    print(f"  - Min: {final_predictions.min():.6f}")
    print(f"  - Max: {final_predictions.max():.6f}")
    
    # Comparaison avec le train
    print(f"\n📊 Comparaison avec le train:")
    print(f"  - Train mean: {y_train.mean():.6f}")
    print(f"  - Train std: {y_train.std():.6f}")
    print(f"  - Ratio std (pred/train): {final_predictions.std() / y_train.std():.3f}")
    
    # 13. Visualisation
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Série temporelle
    plt.subplot(2, 2, 1)
    plt.plot(test_df['Dates'], final_predictions, 'b-', alpha=0.7, label='Prédictions')
    plt.title('Prédictions finales')
    plt.xlabel('Date')
    plt.ylabel('Valeur')
    plt.xticks(rotation=45)
    plt.legend()
    
    # Plot 2: Distribution
    plt.subplot(2, 2, 2)
    plt.hist(final_predictions, bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(final_predictions.mean(), color='red', linestyle='--', label=f'Mean: {final_predictions.mean():.4f}')
    plt.title('Distribution des prédictions')
    plt.xlabel('Valeur')
    plt.ylabel('Fréquence')
    plt.legend()
    
    # Plot 3: Comparaison des distributions (train vs pred)
    plt.subplot(2, 2, 3)
    plt.hist(y_train, bins=50, alpha=0.5, color='blue', label='Train', density=True)
    plt.hist(final_predictions, bins=50, alpha=0.5, color='red', label='Predictions', density=True)
    plt.title('Comparaison des distributions')
    plt.xlabel('Valeur')
    plt.ylabel('Densité')
    plt.legend()
    
    # Plot 4: Rolling mean
    plt.subplot(2, 2, 4)
    rolling_mean = pd.Series(final_predictions).rolling(window=30).mean()
    plt.plot(test_df['Dates'], rolling_mean, 'g-', alpha=0.7)
    plt.title('Moyenne mobile (30 jours)')
    plt.xlabel('Date')
    plt.ylabel('Valeur')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('predictions_ultimate_analysis.png', dpi=300)
    plt.show()
    
    print("\n✅ Analyse complète terminée!")
    print("\n🎯 Cette version Ultimate inclut:")
    print("  - Feature engineering avancé (>1000 features)")
    print("  - Sélection intelligente des features")
    print("  - Optimisation Bayésienne des hyperparamètres")
    print("  - Stacking avancé avec méta-apprentissage")
    print("  - Target transformation et post-processing")
    print("  - Blending de plusieurs approches")

if __name__ == "__main__":
    main()