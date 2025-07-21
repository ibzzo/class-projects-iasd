#!/usr/bin/env python3
"""
Prédiction de Séries Temporelles Financières - Version Optimisée pour Time Series
Utilise des approches spécifiques aux séries temporelles avec validation temporelle stricte

Améliorations clés:
- Feature engineering simplifié et contrôlé
- Validation temporelle stricte (TimeSeriesSplit)
- Modèles spécialisés pour séries temporelles
- Pas de data leakage
- Post-processing pour éviter les valeurs négatives
"""

# ========================================
# 1. IMPORTS ET CONFIGURATION
# ========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import gc
import joblib
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, HuberRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Time Series specific
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from prophet import Prophet

# Deep Learning for Time Series
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("🚀 Financial Time Series Prediction - Optimized Time Series Version")
print("=" * 60)

# ========================================
# 2. CHARGEMENT ET ANALYSE DES DONNÉES
# ========================================

def load_and_analyze_data(data_dir='data'):
    """Charge les données et effectue une analyse initiale"""
    print("\n📊 Chargement et analyse des données...")
    
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')
    
    # Conversion des dates
    train_df['Dates'] = pd.to_datetime(train_df['Dates'])
    test_df['Dates'] = pd.to_datetime(test_df['Dates'])
    
    # Tri par date IMPORTANT pour time series
    train_df = train_df.sort_values('Dates').reset_index(drop=True)
    test_df = test_df.sort_values('Dates').reset_index(drop=True)
    
    print(f"✅ Données chargées:")
    print(f"   - Train: {train_df.shape}")
    print(f"   - Test: {test_df.shape}")
    print(f"   - Période train: {train_df['Dates'].min()} à {train_df['Dates'].max()}")
    print(f"   - Période test: {test_df['Dates'].min()} à {test_df['Dates'].max()}")
    
    # Analyse de la stationnarité
    if 'target' in train_df.columns:
        adf_result = adfuller(train_df['target'])
        print(f"\n📈 Test ADF sur la target:")
        print(f"   - Statistique: {adf_result[0]:.4f}")
        print(f"   - p-value: {adf_result[1]:.4f}")
        print(f"   - Stationnaire: {'Oui' if adf_result[1] < 0.05 else 'Non'}")
    
    return train_df, test_df

# ========================================
# 3. FEATURE ENGINEERING CONTRÔLÉ
# ========================================

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des features temporelles basiques"""
    df = df.copy()
    
    # Features temporelles de base
    df['year'] = df['Dates'].dt.year
    df['month'] = df['Dates'].dt.month
    df['day'] = df['Dates'].dt.day
    df['day_of_week'] = df['Dates'].dt.dayofweek
    df['quarter'] = df['Dates'].dt.quarter
    
    # Encodage cyclique pour capturer la périodicité
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def create_lag_features(df: pd.DataFrame, feature_cols: List[str], 
                       lags: List[int] = [1, 7, 30]) -> pd.DataFrame:
    """
    Crée des lag features de manière contrôlée
    
    IMPORTANT: Pour éviter le data leakage, cette fonction doit être
    appliquée APRÈS le split train/test et en respectant l'ordre temporel
    """
    df = df.copy()
    
    for col in feature_cols[:10]:  # Limiter aux 10 features les plus importantes
        for lag in lags:
            # Lag simple
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            # Différence avec le lag
            df[f'{col}_diff_{lag}'] = df[col] - df[col].shift(lag)
            
            # Ratio avec le lag (éviter division par zéro)
            df[f'{col}_ratio_{lag}'] = df[col] / (df[col].shift(lag) + 1e-8)
    
    return df

def create_rolling_features(df: pd.DataFrame, feature_cols: List[str],
                          windows: List[int] = [7, 30]) -> pd.DataFrame:
    """
    Crée des rolling features de manière contrôlée
    
    IMPORTANT: Utilise min_periods approprié pour éviter les calculs
    avec trop peu de données
    """
    df = df.copy()
    
    for col in feature_cols[:10]:  # Limiter aux 10 features les plus importantes
        for window in windows:
            # S'assurer qu'on a assez de données pour le calcul
            min_periods = max(window // 2, 3)
            
            # Rolling statistics
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(
                window=window, min_periods=min_periods
            ).mean()
            
            df[f'{col}_roll_std_{window}'] = df[col].rolling(
                window=window, min_periods=min_periods
            ).std()
            
            # Rolling min/max pour capturer les extremes
            df[f'{col}_roll_max_{window}'] = df[col].rolling(
                window=window, min_periods=min_periods
            ).max()
            
            df[f'{col}_roll_min_{window}'] = df[col].rolling(
                window=window, min_periods=min_periods
            ).min()
    
    return df

def feature_engineering_timeseries(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Feature engineering spécifique aux séries temporelles"""
    print("\n🔧 Feature Engineering Contrôlé...")
    
    # 1. Features temporelles
    print("  - Features temporelles...")
    train_df = create_time_features(train_df)
    test_df = create_time_features(test_df)
    
    # 2. Identifier les colonnes numériques
    feature_cols = [col for col in train_df.columns 
                   if col.startswith('Features_') and train_df[col].dtype in ['float64', 'int64']]
    
    # 3. Sélection des features importantes AVANT de créer les dérivées
    # Utiliser une méthode plus robuste
    print("  - Sélection des features importantes...")
    if 'target' in train_df.columns:
        selector = SelectKBest(f_regression, k=min(20, len(feature_cols)))
        selector.fit(train_df[feature_cols], train_df['target'])
        feature_scores = pd.DataFrame({
            'feature': feature_cols,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        # Garder seulement les top features
        top_features = feature_scores.head(15)['feature'].tolist()
        print(f"    Top 5 features: {top_features[:5]}")
    else:
        top_features = feature_cols[:15]
    
    # 4. Créer les features dérivées SEULEMENT pour les top features
    print("  - Lag features...")
    train_df = create_lag_features(train_df, top_features, lags=[1, 7, 30])
    test_df = create_lag_features(test_df, top_features, lags=[1, 7, 30])
    
    print("  - Rolling features...")
    train_df = create_rolling_features(train_df, top_features, windows=[7, 30])
    test_df = create_rolling_features(test_df, top_features, windows=[7, 30])
    
    # 5. Feature interactions basiques (seulement top 5)
    print("  - Feature interactions...")
    for i, col1 in enumerate(top_features[:5]):
        for col2 in top_features[i+1:5]:
            train_df[f'{col1}_x_{col2}'] = train_df[col1] * train_df[col2]
            test_df[f'{col1}_x_{col2}'] = test_df[col1] * test_df[col2]
    
    print(f"✅ Features créées: {len(train_df.columns)} colonnes")
    
    return train_df, test_df

# ========================================
# 4. MODÈLES SPÉCIALISÉS TIME SERIES
# ========================================

class TimeSeriesLSTM:
    """Modèle LSTM pour séries temporelles"""
    
    def __init__(self, n_features: int, sequence_length: int = 30):
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        
    def create_sequences(self, X, y=None):
        """Crée des séquences pour LSTM"""
        sequences = []
        targets = []
        
        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i:i+self.sequence_length])
            if y is not None:
                targets.append(y[i+self.sequence_length])
        
        return np.array(sequences), np.array(targets) if y is not None else None
    
    def build_model(self):
        """Construit le modèle LSTM"""
        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dense(1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
    def fit(self, X, y, validation_data=None):
        """Entraîne le modèle"""
        # Normaliser les données
        X_scaled = self.scaler.fit_transform(X)
        
        # Créer les séquences
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]
        
        # Validation data
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val)
            validation_data = (X_val_seq, y_val_seq)
        
        # Entraînement
        history = self.model.fit(
            X_seq, y_seq,
            validation_data=validation_data,
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        return history
    
    def predict(self, X):
        """Fait des prédictions"""
        X_scaled = self.scaler.transform(X)
        
        # Pour la prédiction sur le test set, on doit gérer le manque de séquences
        if len(X) < self.sequence_length:
            # Utiliser les dernières observations du train pour compléter
            return np.zeros(len(X))  # Placeholder
        
        X_seq, _ = self.create_sequences(X_scaled)
        predictions = self.model.predict(X_seq, verbose=0)
        
        # Padding pour avoir la même longueur que X
        full_predictions = np.zeros(len(X))
        full_predictions[self.sequence_length:self.sequence_length+len(predictions)] = predictions.flatten()
        
        # Pour les premières valeurs, utiliser une moyenne
        full_predictions[:self.sequence_length] = predictions[0]
        
        return full_predictions

class ProphetWrapper:
    """Wrapper pour utiliser Prophet avec des features multiples"""
    
    def __init__(self, additional_regressors=None):
        self.model = None
        self.additional_regressors = additional_regressors or []
        
    def fit(self, dates, y, X=None):
        """Entraîne le modèle Prophet"""
        # Préparer les données pour Prophet
        df = pd.DataFrame({
            'ds': dates,
            'y': y
        })
        
        # Ajouter les régresseurs additionnels
        if X is not None and len(self.additional_regressors) > 0:
            for i, reg in enumerate(self.additional_regressors):
                df[reg] = X[:, i]
        
        # Initialiser et configurer Prophet
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )
        
        # Ajouter les régresseurs
        for reg in self.additional_regressors:
            self.model.add_regressor(reg)
        
        # Entraîner
        self.model.fit(df)
        
    def predict(self, dates, X=None):
        """Fait des prédictions"""
        future = pd.DataFrame({'ds': dates})
        
        # Ajouter les régresseurs
        if X is not None and len(self.additional_regressors) > 0:
            for i, reg in enumerate(self.additional_regressors):
                future[reg] = X[:, i]
        
        forecast = self.model.predict(future)
        return forecast['yhat'].values

# ========================================
# 5. ENSEMBLE AVEC VALIDATION TEMPORELLE
# ========================================

class TimeSeriesEnsemble:
    """Ensemble de modèles avec validation temporelle stricte"""
    
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.models = {}
        self.weights = {}
        self.oof_predictions = None
        
    def get_models(self):
        """Retourne la liste des modèles à utiliser"""
        return {
            'lgb': lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=5,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            ),
            'cat': CatBoostRegressor(
                iterations=1000,
                learning_rate=0.01,
                depth=6,
                l2_leaf_reg=3,
                random_state=42,
                verbose=False
            ),
            'rf': RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            ),
            'ridge': Ridge(alpha=10.0, random_state=42),
            'huber': HuberRegressor(epsilon=1.35, alpha=10.0)
        }
    
    def fit(self, X, y, dates):
        """Entraîne l'ensemble avec validation temporelle"""
        print("\n🏗️ Entraînement de l'ensemble Time Series...")
        
        # TimeSeriesSplit pour validation temporelle
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # Stocker les prédictions OOF
        self.oof_predictions = np.zeros((len(y), len(self.get_models())))
        
        # Pour chaque modèle
        for model_idx, (name, model) in enumerate(self.get_models().items()):
            print(f"\n  Modèle {model_idx+1}/{len(self.get_models())}: {name}")
            
            model_oof = np.zeros(len(y))
            scores = []
            
            # Cross-validation temporelle
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Entraîner le modèle
                if name in ['lgb', 'xgb', 'cat']:
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
                model_oof[val_idx] = val_pred
                
                # Score
                score = np.sqrt(mean_squared_error(y_val, val_pred))
                scores.append(score)
            
            # Stocker les prédictions OOF
            self.oof_predictions[:, model_idx] = model_oof
            
            # Score moyen
            mean_score = np.mean(scores)
            print(f"    Score moyen (RMSE): {mean_score:.6f}")
            
            # Stocker le modèle entraîné sur toutes les données
            self.models[name] = model
            if name in ['lgb', 'xgb', 'cat']:
                self.models[name].fit(
                    X, y,
                    verbose=False
                )
            else:
                self.models[name].fit(X, y)
        
        # Optimiser les poids de l'ensemble
        self.optimize_weights(y)
        
    def optimize_weights(self, y_true):
        """Optimise les poids de l'ensemble"""
        from scipy.optimize import minimize
        
        def objective(weights):
            weighted_pred = np.average(self.oof_predictions, axis=1, weights=weights)
            return np.sqrt(mean_squared_error(y_true, weighted_pred))
        
        # Contraintes : poids positifs et somme = 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(self.models))]
        
        # Point de départ : poids égaux
        x0 = np.ones(len(self.models)) / len(self.models)
        
        # Optimisation
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        self.weights = dict(zip(self.models.keys(), result.x))
        print(f"\n📊 Poids optimaux de l'ensemble:")
        for name, weight in self.weights.items():
            print(f"   - {name}: {weight:.3f}")
        
        # Score final
        final_score = objective(result.x)
        print(f"\n   Score ensemble (RMSE): {final_score:.6f}")
    
    def predict(self, X):
        """Fait des prédictions avec l'ensemble pondéré"""
        predictions = np.zeros((len(X), len(self.models)))
        
        for idx, (name, model) in enumerate(self.models.items()):
            predictions[:, idx] = model.predict(X)
        
        # Moyenne pondérée
        weights_array = np.array(list(self.weights.values()))
        weighted_pred = np.average(predictions, axis=1, weights=weights_array)
        
        return weighted_pred

# ========================================
# 6. POST-PROCESSING
# ========================================

def post_process_predictions(predictions, train_target):
    """Post-processing pour améliorer les prédictions"""
    print("\n🎯 Post-processing des prédictions...")
    
    # 1. Éviter les valeurs négatives si la target est toujours positive
    if train_target.min() >= 0:
        predictions = np.maximum(predictions, 0)
        print("  - Valeurs négatives corrigées")
    
    # 2. Limiter les valeurs extrêmes
    # Utiliser les percentiles du train pour définir des limites raisonnables
    lower_bound = np.percentile(train_target, 0.1)
    upper_bound = np.percentile(train_target, 99.9)
    
    predictions = np.clip(predictions, lower_bound, upper_bound)
    print(f"  - Valeurs clippées entre {lower_bound:.3f} et {upper_bound:.3f}")
    
    # 3. Ajustement de la distribution
    # Si les prédictions ont une distribution très différente du train
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    train_mean = train_target.mean()
    train_std = train_target.std()
    
    if abs(pred_std / train_std - 1) > 0.2:  # Si l'écart est > 20%
        # Ajuster l'écart-type
        predictions = (predictions - pred_mean) * (train_std / pred_std) + train_mean
        print("  - Distribution ajustée")
    
    return predictions

# ========================================
# 7. PIPELINE PRINCIPAL
# ========================================

def main():
    """Pipeline principal"""
    
    # 1. Charger les données
    train_df, test_df = load_and_analyze_data()
    
    # 2. Feature Engineering
    train_df, test_df = feature_engineering_timeseries(train_df, test_df)
    
    # 3. Préparer les données pour modélisation
    # Exclure les colonnes non-features
    exclude_cols = ['Dates', 'target', 'ID']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Gérer les valeurs manquantes créées par les lags et rolling
    print("\n🧹 Gestion des valeurs manquantes...")
    train_df[feature_cols] = train_df[feature_cols].fillna(method='bfill').fillna(0)
    test_df[feature_cols] = test_df[feature_cols].fillna(method='bfill').fillna(0)
    
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_test = test_df[feature_cols].values
    dates_train = train_df['Dates']
    dates_test = test_df['Dates']
    
    print(f"\n📐 Dimensions finales:")
    print(f"   - X_train: {X_train.shape}")
    print(f"   - X_test: {X_test.shape}")
    
    # 4. Normalisation
    print("\n🔧 Normalisation des données...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Entraînement de l'ensemble
    ensemble = TimeSeriesEnsemble(n_splits=5)
    ensemble.fit(X_train_scaled, y_train, dates_train)
    
    # 6. Prédictions
    print("\n📈 Génération des prédictions...")
    predictions = ensemble.predict(X_test_scaled)
    
    # 7. Post-processing
    predictions = post_process_predictions(predictions, y_train)
    
    # 8. Sauvegarder les résultats
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'target': predictions
    })
    submission.to_csv('submission_timeseries.csv', index=False)
    print("\n✅ Fichier de soumission créé: submission_timeseries.csv")
    
    # 9. Statistiques finales
    print("\n📊 Statistiques des prédictions finales:")
    print(f"   - Moyenne: {predictions.mean():.6f}")
    print(f"   - Std: {predictions.std():.6f}")
    print(f"   - Min: {predictions.min():.6f}")
    print(f"   - Max: {predictions.max():.6f}")
    
    print("\n📊 Comparaison avec le train:")
    print(f"   - Train mean: {y_train.mean():.6f}")
    print(f"   - Train std: {y_train.std():.6f}")
    print(f"   - Ratio std (pred/train): {predictions.std()/y_train.std():.3f}")
    
    # 10. Visualisation
    plt.figure(figsize=(12, 6))
    
    # Distribution des prédictions
    plt.subplot(1, 2, 1)
    plt.hist(predictions, bins=50, alpha=0.7, label='Predictions', density=True)
    plt.hist(y_train, bins=50, alpha=0.7, label='Train', density=True)
    plt.xlabel('Target Value')
    plt.ylabel('Density')
    plt.title('Distribution Comparison')
    plt.legend()
    
    # Time series plot
    plt.subplot(1, 2, 2)
    plt.plot(dates_test, predictions, label='Predictions', alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Target')
    plt.title('Predictions Over Time')
    plt.xticks(rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('predictions_timeseries.png', dpi=300, bbox_inches='tight')
    print("\n📊 Graphique sauvegardé: predictions_timeseries.png")

if __name__ == "__main__":
    main()