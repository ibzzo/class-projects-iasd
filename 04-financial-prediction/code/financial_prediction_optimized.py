#!/usr/bin/env python3
"""
Prédiction de Séries Temporelles Financières - Version Optimisée
Kaggle Competition pour IASD Master 2

Ce script implémente une solution complète pour la prédiction de séries temporelles financières
avec des techniques avancées pour éviter le surapprentissage.
"""

# ========================================
# 1. IMPORTS ET CONFIGURATION
# ========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import platform
import subprocess
import sys

warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import PolynomialFeatures

# Deep Learning avec PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Advanced ML
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM non disponible, utilisation de XGBoost comme alternative")

import xgboost as xgb
from catboost import CatBoostRegressor

# Hyperparameter Optimization
import optuna

# Interpretability
import shap

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ========================================
# 2. CONFIGURATION POUR MAC (LIBOMP)
# ========================================

def setup_mac_environment():
    """Configure l'environnement pour Mac (nécessaire pour LightGBM)"""
    if platform.system() == 'Darwin':
        print("🍎 Configuration pour macOS détectée")
        
        # Vérifier si libomp est installé
        libomp_paths = [
            '/opt/homebrew/opt/libomp/lib/libomp.dylib',  # Apple Silicon
            '/usr/local/opt/libomp/lib/libomp.dylib',     # Intel Mac
        ]
        
        libomp_found = None
        for path in libomp_paths:
            if os.path.exists(path):
                libomp_found = path
                break
        
        if libomp_found:
            print(f"✅ libomp trouvé à: {libomp_found}")
            # Ajouter au DYLD_LIBRARY_PATH
            lib_dir = os.path.dirname(libomp_found)
            os.environ['DYLD_LIBRARY_PATH'] = f"{lib_dir}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"
        else:
            print("⚠️ libomp non trouvé. Installez avec: brew install libomp")
            return False
    return True

# ========================================
# 3. CHARGEMENT DES DONNÉES
# ========================================

def load_data(data_dir='data'):
    """Charge les données d'entraînement et de test"""
    print("📊 Chargement des données...")
    
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')
    
    # Essayer de charger les descriptions des colonnes
    try:
        # Essayer d'abord le fichier corrigé
        if os.path.exists(f'{data_dir}/column_description_fixed.csv'):
            column_desc = pd.read_csv(f'{data_dir}/column_description_fixed.csv', 
                                    header=None, names=['column', 'description'])
        else:
            # Sinon, lire le fichier original ligne par ligne
            with open(f'{data_dir}/column_description.csv', 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            column_desc_data = []
            for line in lines:
                line = line.strip()
                if line and ',' in line:
                    first_comma = line.find(',')
                    column_name = line[:first_comma].strip()
                    description = line[first_comma+1:].strip()
                    column_desc_data.append({'column': column_name, 'description': description})
            
            column_desc = pd.DataFrame(column_desc_data)
        
        print(f"✅ {len(column_desc)} descriptions de colonnes chargées")
    except Exception as e:
        print(f"⚠️ Impossible de charger les descriptions: {e}")
        column_desc = None
    
    # Conversion des dates
    train_df['Dates'] = pd.to_datetime(train_df['Dates'])
    test_df['Dates'] = pd.to_datetime(test_df['Dates'])
    
    # Tri par date
    train_df = train_df.sort_values('Dates').reset_index(drop=True)
    test_df = test_df.sort_values('Dates').reset_index(drop=True)
    
    print(f"✅ Données chargées:")
    print(f"   - Train: {train_df.shape}")
    print(f"   - Test: {test_df.shape}")
    print(f"   - Période train: {train_df['Dates'].min()} à {train_df['Dates'].max()}")
    print(f"   - Période test: {test_df['Dates'].min()} à {test_df['Dates'].max()}")
    
    return train_df, test_df, column_desc

# ========================================
# 4. FEATURE ENGINEERING
# ========================================

def create_advanced_features(df, is_train=True):
    """
    Crée des features avancées incluant:
    - Transformations non-linéaires
    - Features de régime de marché
    - Statistiques glissantes
    - Interactions entre features
    """
    print("\n🔧 Feature Engineering...")
    df_feat = df.copy()
    
    # Obtenir les colonnes de features
    feature_cols = [col for col in df.columns if col.startswith('Features_')]
    
    # 1. Transformations non-linéaires (top 20 features)
    print("  - Création des transformations non-linéaires...")
    for col in feature_cols[:20]:
        # Log transformation
        df_feat[f'{col}_log'] = np.log1p(df_feat[col] - df_feat[col].min() + 1)
        # Square root transformation
        df_feat[f'{col}_sqrt'] = np.sqrt(np.abs(df_feat[col]))
        # Power transformation
        df_feat[f'{col}_squared'] = df_feat[col] ** 2
        # Exponential transformation (normalized)
        df_feat[f'{col}_exp'] = np.exp(df_feat[col] / df_feat[col].std() if df_feat[col].std() > 0 else 1)
    
    # 2. Features de volatilité et régime
    print("  - Création des features de volatilité...")
    volatility_windows = [5, 10, 20, 30, 60]
    for window in volatility_windows:
        for col in feature_cols[:10]:
            df_feat[f'{col}_volatility_{window}'] = df_feat[col].rolling(window).std()
            df_feat[f'{col}_mean_{window}'] = df_feat[col].rolling(window).mean()
            df_feat[f'{col}_skew_{window}'] = df_feat[col].rolling(window).skew()
            df_feat[f'{col}_kurt_{window}'] = df_feat[col].rolling(window).kurt()
    
    # 3. Détection de régimes de marché
    print("  - Détection des régimes de marché...")
    regime_features = ['Features_34', 'Features_35', 'Features_36',
                      'Features_79', 'Features_85', 'Features_93']
    
    if all(col in df_feat.columns for col in regime_features):
        regime_data = df_feat[regime_features].fillna(method='ffill').fillna(0)
        
        n_regimes = 5
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=20)
        df_feat['market_regime'] = kmeans.fit_predict(regime_data)
        
        for regime in range(n_regimes):
            df_feat[f'is_regime_{regime}'] = (df_feat['market_regime'] == regime).astype(int)
    
    # 4. Volatilité conditionnelle (style GARCH)
    print("  - Création des features de volatilité conditionnelle...")
    for col in feature_cols[:3]:
        returns = df_feat[col].pct_change()
        squared_returns = returns ** 2
        df_feat[f'{col}_cond_vol'] = squared_returns.ewm(span=10, adjust=False).mean()
    
    # 5. Interactions polynomiales et ratios
    print("  - Création des interactions entre features...")
    top_features = feature_cols[:10]
    
    # Interactions polynomiales
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_features = poly.fit_transform(df_feat[top_features].fillna(0))
    
    poly_feature_names = poly.get_feature_names_out(top_features)
    interaction_mask = ['*' in name for name in poly_feature_names]
    interaction_features = poly_features[:, interaction_mask]
    interaction_names = [name for name, mask in zip(poly_feature_names, interaction_mask) if mask]
    
    for i, name in enumerate(interaction_names[:20]):
        df_feat[f'interaction_{name}'] = interaction_features[:, i]
    
    # Ratios entre features importantes
    for i in range(min(5, len(feature_cols))):
        for j in range(i+1, min(10, len(feature_cols))):
            ratio_name = f'ratio_{feature_cols[i]}_{feature_cols[j]}'
            df_feat[ratio_name] = df_feat[feature_cols[i]] / (df_feat[feature_cols[j]] + 1e-8)
    
    # 6. Features temporelles
    print("  - Création des features temporelles...")
    df_feat['year'] = pd.to_datetime(df_feat['Dates']).dt.year
    df_feat['month'] = pd.to_datetime(df_feat['Dates']).dt.month
    df_feat['day_of_year'] = pd.to_datetime(df_feat['Dates']).dt.dayofyear
    df_feat['quarter'] = pd.to_datetime(df_feat['Dates']).dt.quarter
    
    # Remplir les valeurs manquantes
    df_feat = df_feat.fillna(method='ffill').fillna(0)
    
    print(f"✅ Features créées: {df_feat.shape[1]} colonnes")
    
    return df_feat

# ========================================
# 5. SÉLECTION DE FEATURES
# ========================================

class TimeSeriesFeatureSelector:
    """Sélection de features avec validation temporelle"""
    
    def __init__(self, n_splits=5, test_size=0.2):
        self.n_splits = n_splits
        self.test_size = test_size
        self.selected_features = None
        self.feature_importances = None
        
    def select_features(self, X, y, method='permutation', n_features=80):
        """Sélectionne les meilleures features"""
        print(f"\n🎯 Sélection de {n_features} features (méthode: {method})...")
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=int(len(X) * self.test_size))
        
        # Choisir le modèle (LightGBM ou XGBoost)
        use_lgb = LIGHTGBM_AVAILABLE
        
        if use_lgb:
            try:
                # Test rapide de LightGBM
                test_model = lgb.LGBMRegressor(n_estimators=5)
                test_model.fit(X.iloc[:100], y.iloc[:100])
                print("  ✅ Utilisation de LightGBM")
            except Exception as e:
                print(f"  ⚠️ Erreur LightGBM: {str(e)[:50]}...")
                print("  ✅ Basculement sur XGBoost")
                use_lgb = False
        
        if method == 'permutation':
            if use_lgb:
                model = lgb.LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    min_child_samples=20,
                    min_split_gain=0.01,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    n_jobs=1,
                    force_col_wise=True
                )
            else:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    n_jobs=1
                )
            
            print("  - Entraînement du modèle...")
            model.fit(X, y)
            
            print("  - Calcul de l'importance par permutation...")
            perm_importance = permutation_importance(
                model, X, y, 
                n_repeats=5,
                random_state=42,
                n_jobs=1
            )
            
            # Sélectionner les top features
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': perm_importance.importances_mean,
                'std': perm_importance.importances_std
            }).sort_values('importance', ascending=False)
            
            self.selected_features = importance_df.head(n_features)['feature'].tolist()
            self.feature_importances = importance_df
            
        print(f"✅ {len(self.selected_features)} features sélectionnées")
        return self.selected_features
    
    def plot_feature_importance(self, top_n=30):
        """Affiche l'importance des features"""
        if self.feature_importances is not None:
            plt.figure(figsize=(10, 8))
            top_features = self.feature_importances.head(top_n)
            
            plt.barh(top_features['feature'], top_features['importance'])
            plt.xlabel('Importance par permutation')
            plt.title(f'Top {top_n} Features')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

# ========================================
# 6. VALIDATION TEMPORELLE
# ========================================

class PurgedTimeSeriesSplit:
    """Time series cross-validation avec purging et embargo"""
    
    def __init__(self, n_splits=5, test_size=None, gap=0, purge_days=0, embargo_days=0):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.purge_days = purge_days
        self.embargo_days = embargo_days
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
            
        for i in range(self.n_splits):
            train_end = int((i + 1) * (n_samples - test_size) / self.n_splits)
            test_start = train_end + self.gap
            test_end = min(test_start + test_size, n_samples)
            
            if self.purge_days > 0:
                train_end = max(0, train_end - self.purge_days)
            
            if self.embargo_days > 0:
                test_start = min(n_samples, test_start + self.embargo_days)
            
            if train_end > 0 and test_start < test_end:
                yield indices[:train_end], indices[test_start:test_end]

# ========================================
# 7. MODÈLES DEEP LEARNING (PYTORCH)
# ========================================

# Configuration du device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\n🔧 Device PyTorch: {device}")

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length=20):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X) - self.sequence_length
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.sequence_length]
        y_seq = self.y[idx + self.sequence_length]
        return torch.FloatTensor(X_seq), torch.FloatTensor([y_seq])

class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super(LSTMWithAttention, self).__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        out = self.dropout(context)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        return out

# ========================================
# 8. OPTIMISATION BAYÉSIENNE
# ========================================

class BayesianOptimizer:
    """Optimisation bayésienne des hyperparamètres"""
    
    def __init__(self, model_type='lgb', cv_strategy=None):
        self.model_type = model_type
        self.cv_strategy = cv_strategy or PurgedTimeSeriesSplit(n_splits=5)
        self.best_params = None
        self.study = None
        
    def objective(self, trial, X, y):
        """Fonction objectif pour l'optimisation"""
        if self.model_type == 'lgb' and LIGHTGBM_AVAILABLE:
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'n_estimators': 5000,
                'random_state': 42,
                'verbosity': -1
            }
            model_class = lgb.LGBMRegressor
        else:
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': 5000,
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42
            }
            model_class = xgb.XGBRegressor
        
        scores = []
        for train_idx, val_idx in self.cv_strategy.split(X):
            X_train_cv = X.iloc[train_idx]
            X_val_cv = X.iloc[val_idx]
            y_train_cv = y.iloc[train_idx]
            y_val_cv = y.iloc[val_idx]
            
            model = model_class(**params)
            
            if isinstance(model, lgb.LGBMRegressor) and LIGHTGBM_AVAILABLE:
                model.fit(
                    X_train_cv, y_train_cv,
                    eval_set=[(X_val_cv, y_val_cv)],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
            elif hasattr(model, 'fit') and 'eval_set' in model.fit.__code__.co_varnames:
                model.fit(
                    X_train_cv, y_train_cv,
                    eval_set=[(X_val_cv, y_val_cv)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                model.fit(X_train_cv, y_train_cv)
            
            y_pred = model.predict(X_val_cv)
            rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred))
            scores.append(rmse)
        
        return np.mean(scores)
    
    def optimize(self, X, y, n_trials=50):
        """Lance l'optimisation"""
        print(f"\n🔍 Optimisation Bayésienne ({n_trials} essais)...")
        
        self.study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        self.study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=n_trials,
            n_jobs=1,
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        print(f"✅ Meilleurs paramètres trouvés (score: {self.study.best_value:.6f})")
        
        return self.best_params

# ========================================
# 9. MODÈLE D'ENSEMBLE (STACKING)
# ========================================

class AdvancedStackingRegressor:
    """Ensemble stacking avec plusieurs modèles de base"""
    
    def __init__(self, base_models, meta_model, cv_strategy=None):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv_strategy = cv_strategy or PurgedTimeSeriesSplit(n_splits=5)
        self.trained_base_models = []
        
    def fit(self, X, y):
        """Entraîne le modèle stacking"""
        print("\n🏗️ Entraînement du modèle Stacking...")
        
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        # Prédictions out-of-fold
        oof_predictions = np.zeros((n_samples, n_models))
        
        # Entraîner les modèles de base
        for i, (name, model) in enumerate(self.base_models):
            print(f"  - Entraînement {name}...")
            
            for train_idx, val_idx in self.cv_strategy.split(X):
                X_train_cv = X.iloc[train_idx]
                X_val_cv = X.iloc[val_idx]
                y_train_cv = y.iloc[train_idx]
                
                model_clone = model.__class__(**model.get_params())
                
                if isinstance(model_clone, lgb.LGBMRegressor) and LIGHTGBM_AVAILABLE:
                    model_clone.fit(
                        X_train_cv, y_train_cv,
                        eval_set=[(X_val_cv, y.iloc[val_idx])],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                    )
                elif hasattr(model_clone, 'fit') and 'eval_set' in model_clone.fit.__code__.co_varnames:
                    model_clone.fit(
                        X_train_cv, y_train_cv,
                        eval_set=[(X_val_cv, y.iloc[val_idx])],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                else:
                    model_clone.fit(X_train_cv, y_train_cv)
                
                oof_predictions[val_idx, i] = model_clone.predict(X_val_cv)
        
        # Entraîner le meta-modèle
        print("  - Entraînement du meta-modèle...")
        self.meta_model.fit(oof_predictions, y)
        
        # Réentraîner les modèles de base sur toutes les données
        print("  - Réentraînement sur l'ensemble complet...")
        self.trained_base_models = []
        
        for name, model in self.base_models:
            model_clone = model.__class__(**model.get_params())
            
            if isinstance(model_clone, lgb.LGBMRegressor) and LIGHTGBM_AVAILABLE:
                val_size = int(0.1 * len(X))
                X_train_full = X.iloc[:-val_size]
                y_train_full = y.iloc[:-val_size]
                X_val_full = X.iloc[-val_size:]
                y_val_full = y.iloc[-val_size:]
                
                model_clone.fit(
                    X_train_full, y_train_full,
                    eval_set=[(X_val_full, y_val_full)],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
            else:
                model_clone.fit(X, y)
                
            self.trained_base_models.append((name, model_clone))
        
        print("✅ Modèle Stacking entraîné")
        return self
    
    def predict(self, X):
        """Fait des prédictions"""
        n_models = len(self.trained_base_models)
        base_predictions = np.zeros((X.shape[0], n_models))
        
        for i, (name, model) in enumerate(self.trained_base_models):
            base_predictions[:, i] = model.predict(X)
        
        return self.meta_model.predict(base_predictions)

# ========================================
# 10. FONCTION PRINCIPALE
# ========================================

def main():
    """Fonction principale d'exécution"""
    print("=" * 60)
    print("🚀 PRÉDICTION DE SÉRIES TEMPORELLES FINANCIÈRES")
    print("=" * 60)
    
    # Configuration Mac
    setup_mac_environment()
    
    # 1. Chargement des données
    train_df, test_df, column_desc = load_data()
    
    # 2. Feature Engineering
    train_enhanced = create_advanced_features(train_df)
    test_enhanced = create_advanced_features(test_df, is_train=False)
    
    # 3. Sélection de features
    feature_cols = [col for col in train_enhanced.columns 
                    if col not in ['Dates', 'ToPredict']]
    
    X_train = train_enhanced[feature_cols]
    y_train = train_enhanced['ToPredict']
    
    selector = TimeSeriesFeatureSelector(n_splits=5)
    selected_features = selector.select_features(X_train, y_train, n_features=150)
    
    # Afficher l'importance des features
    selector.plot_feature_importance()
    
    # 4. Préparation des données sélectionnées
    X_selected = train_enhanced[selected_features]
    y = train_enhanced['ToPredict']
    X_test = test_enhanced[selected_features]
    
    # 5. Optimisation Bayésienne
    print("\n🔍 Optimisation Bayésienne en cours...")
    optimizer = BayesianOptimizer(model_type='xgb')
    best_params = optimizer.optimize(X_selected, y, n_trials=100)
    
    # 6. Définition des modèles de base
    base_models = []
    
    # LightGBM (si disponible)
    if LIGHTGBM_AVAILABLE:
        base_models.append(('LightGBM', lgb.LGBMRegressor(
            n_estimators=5000,
            learning_rate=0.03,
            max_depth=8,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.05,
            num_leaves=127,
            min_split_gain=0.001,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )))
    
    # XGBoost
    base_models.append(('XGBoost', xgb.XGBRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.05,
        gamma=0.01,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )))
    
    # CatBoost
    base_models.append(('CatBoost', CatBoostRegressor(
        iterations=5000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=2,
        random_state=42,
        verbose=False,
        thread_count=-1,
        task_type='CPU'
    )))
    
    # Modèles linéaires
    base_models.extend([
        ('Ridge', Ridge(alpha=1.0, random_state=42)),
        ('ElasticNet', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42))
    ])
    
    # Ajouter plus de modèles
    base_models.extend([
        ('GradientBoosting', GradientBoostingRegressor(
            n_estimators=3000,
            learning_rate=0.03,
            max_depth=7,
            min_samples_split=20,
            min_samples_leaf=15,
            subsample=0.85,
            random_state=42
        )),
        ('RandomForest', RandomForestRegressor(
            n_estimators=2000,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )),
        ('ExtraGradientBoosting', GradientBoostingRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=30,
            min_samples_leaf=20,
            subsample=0.8,
            loss='huber',
            alpha=0.9,
            random_state=42
        ))
    ])
    
    # Meta-modèle plus sophistiqué
    meta_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    # 7. Entraînement du modèle Stacking
    stacking_model = AdvancedStackingRegressor(base_models, meta_model)
    stacking_model.fit(X_selected, y)
    
    # 8. Prédictions
    print("\n📈 Génération des prédictions...")
    test_predictions = stacking_model.predict(X_test)
    
    # 9. Création du fichier de soumission
    submission = pd.DataFrame({
        'ID': test_df['Dates'].dt.strftime('%Y-%m-%d'),
        'ToPredict': test_predictions
    })
    
    submission.to_csv('submission_optimized.csv', index=False)
    print("✅ Fichier de soumission créé: submission_optimized.csv")
    
    # 10. Statistiques finales
    print("\n📊 Statistiques des prédictions:")
    print(submission['ToPredict'].describe())
    
    # Visualisation
    plt.figure(figsize=(12, 5))
    plt.plot(test_df['Dates'], test_predictions)
    plt.title('Prédictions sur l\'ensemble de test')
    plt.xlabel('Date')
    plt.ylabel('Valeur prédite')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('predictions_plot.png')
    plt.show()
    
    print("\n✅ Analyse terminée avec succès!")

# ========================================
# EXÉCUTION
# ========================================

if __name__ == "__main__":
    main()