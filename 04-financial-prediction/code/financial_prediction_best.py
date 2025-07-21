#!/usr/bin/env python3
"""
Prédiction de Séries Temporelles Financières - Version Best Performance
Basé sur l'approche qui a donné les meilleurs résultats dans le notebook

Cette version privilégie la simplicité et l'efficacité avec :
- Feature engineering extensif mais ciblé
- Modèles simples mais robustes (RandomForest a donné le meilleur score)
- Ensemble pondéré basé sur les performances
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
from typing import List, Tuple, Dict

warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Set random seeds
np.random.seed(42)

print("🚀 Financial Time Series Prediction - Best Performance Version")
print("=" * 60)

# ========================================
# 2. CHARGEMENT DES DONNÉES
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
    print(f"   - Période train: {train_df['Dates'].min()} à {train_df['Dates'].max()}")
    print(f"   - Période test: {test_df['Dates'].min()} à {test_df['Dates'].max()}")
    
    return train_df, test_df

# ========================================
# 3. FEATURE ENGINEERING COMPLET
# ========================================

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée les features temporelles avec encodage cyclique"""
    print("  - Création des features temporelles...")
    
    # Features temporelles de base
    df['year'] = df['Dates'].dt.year
    df['month'] = df['Dates'].dt.month
    df['day'] = df['Dates'].dt.day
    df['day_of_week'] = df['Dates'].dt.dayofweek
    df['day_of_year'] = df['Dates'].dt.dayofyear
    df['quarter'] = df['Dates'].dt.quarter
    df['week_of_year'] = df['Dates'].dt.isocalendar().week
    
    # Encodage cyclique pour capturer la périodicité
    # Mois
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Jour du mois
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    # Jour de la semaine
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def create_rolling_features(df: pd.DataFrame, feature_cols: List[str], 
                          windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
    """Crée des features de rolling window statistics"""
    print("  - Création des rolling features...")
    
    # Sélectionner les top 10 features pour les rolling stats
    top_features = feature_cols[:10]
    
    for window in windows:
        for col in top_features:
            # Statistiques de base
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
            df[f'{col}_roll_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
            df[f'{col}_roll_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
    
    return df

def create_lag_features(df: pd.DataFrame, feature_cols: List[str], 
                       lags: List[int] = [1, 2, 3, 5, 10, 20]) -> pd.DataFrame:
    """Crée des features lag"""
    print("  - Création des lag features...")
    
    # Top 5 features pour les lags
    top_features = feature_cols[:5]
    
    for lag in lags:
        for col in top_features:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df

def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calcule le Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / (loss + 1e-10)  # Éviter la division par zéro
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

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

def create_technical_indicators(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Crée des indicateurs techniques inspirés de la finance"""
    print("  - Création des indicateurs techniques...")
    
    # Top 5 features pour les indicateurs techniques
    top_features = feature_cols[:5]
    
    for col in top_features:
        # RSI
        df[f'{col}_rsi'] = calculate_rsi(df[col])
        
        # Bollinger Bands
        bb = calculate_bollinger_bands(df[col])
        df[f'{col}_bb_upper'] = bb['upper']
        df[f'{col}_bb_lower'] = bb['lower']
        df[f'{col}_bb_width'] = bb['width']
        df[f'{col}_bb_position'] = bb['position']
    
    return df

def engineer_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Pipeline complet de feature engineering"""
    print(f"\n🔧 Feature Engineering {'(Train)' if is_train else '(Test)'}...")
    
    df_feat = df.copy()
    
    # Obtenir les colonnes de features
    feature_cols = [col for col in df.columns if col.startswith('Features_')]
    
    # 1. Features temporelles avec encodage cyclique
    df_feat = create_temporal_features(df_feat)
    
    # 2. Rolling window features
    df_feat = create_rolling_features(df_feat, feature_cols)
    
    # 3. Lag features
    df_feat = create_lag_features(df_feat, feature_cols)
    
    # 4. Indicateurs techniques
    df_feat = create_technical_indicators(df_feat, feature_cols)
    
    # 5. Traitement des valeurs manquantes
    # Forward fill puis backward fill pour les séries temporelles
    df_feat = df_feat.fillna(method='ffill').fillna(method='bfill')
    
    # Remplir les valeurs restantes avec 0
    df_feat = df_feat.fillna(0)
    
    print(f"✅ Features créées: {df_feat.shape[1]} colonnes (vs {df.shape[1]} originales)")
    
    return df_feat

# ========================================
# 4. MODÈLES ET VALIDATION
# ========================================

def get_models() -> Dict:
    """Retourne les modèles à tester avec leurs paramètres optimaux"""
    models = {
        # Le meilleur modèle selon les résultats
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        
        # Autres bons modèles
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        ),
        
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1
        ),
        
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ),
        
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        ),
        
        'CatBoost': CatBoostRegressor(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3,
            random_state=42,
            verbose=False,
            thread_count=-1
        ),
        
        # Modèles linéaires
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.01, random_state=42, max_iter=2000),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=2000),
        'Huber': HuberRegressor(epsilon=1.35, max_iter=1000)
    }
    
    return models

def validate_models(X: pd.DataFrame, y: pd.Series, models: Dict, cv_splits: int = 5) -> pd.DataFrame:
    """Valide les modèles avec TimeSeriesSplit"""
    print("\n📊 Validation des modèles...")
    
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    results = []
    
    for name, model in models.items():
        print(f"\n  - Validation {name}...")
        
        # Cross-validation
        cv_results = cross_validate(
            model, X, y,
            cv=tscv,
            scoring=['neg_mean_squared_error', 'neg_mean_absolute_error'],
            return_train_score=True,
            n_jobs=1  # Éviter les problèmes de parallélisation
        )
        
        # Calculer les métriques
        train_mse = -cv_results['train_neg_mean_squared_error'].mean()
        val_mse = -cv_results['test_neg_mean_squared_error'].mean()
        val_mae = -cv_results['test_neg_mean_absolute_error'].mean()
        val_rmse = np.sqrt(val_mse)
        
        # Ratio de surapprentissage
        overfit_ratio = (val_mse / train_mse - 1) * 100 if train_mse > 0 else 0
        
        results.append({
            'Model': name,
            'Train_MSE': train_mse,
            'Val_MSE': val_mse,
            'Val_RMSE': val_rmse,
            'Val_MAE': val_mae,
            'Overfit_Ratio': overfit_ratio
        })
        
        print(f"    Val MSE: {val_mse:.6f}, RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}")
        print(f"    Overfit Ratio: {overfit_ratio:.2f}%")
    
    results_df = pd.DataFrame(results).sort_values('Val_MSE')
    return results_df

# ========================================
# 5. ENSEMBLE ET PRÉDICTIONS
# ========================================

def create_weighted_ensemble(models: Dict, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, results_df: pd.DataFrame,
                           top_n: int = 3) -> np.ndarray:
    """Crée un ensemble pondéré des meilleurs modèles"""
    print(f"\n🎯 Création de l'ensemble pondéré (top {top_n} modèles)...")
    
    # Sélectionner les top N modèles
    top_models = results_df.head(top_n)['Model'].tolist()
    
    # Calculer les poids basés sur l'inverse du MSE
    mse_values = results_df[results_df['Model'].isin(top_models)]['Val_MSE'].values
    weights = 1 / mse_values
    weights = weights / weights.sum()
    
    print("\n📊 Poids de l'ensemble:")
    for model, weight in zip(top_models, weights):
        print(f"  - {model}: {weight:.2%}")
    
    # Entraîner les modèles sur l'ensemble complet et faire les prédictions
    predictions = []
    
    for model_name in top_models:
        print(f"\n  - Entraînement {model_name} sur l'ensemble complet...")
        model = models[model_name]
        
        # Entraîner sur toutes les données
        model.fit(X_train, y_train)
        
        # Prédire
        pred = model.predict(X_test)
        predictions.append(pred)
    
    # Ensemble pondéré
    predictions = np.array(predictions)
    final_predictions = np.average(predictions, axis=0, weights=weights)
    
    # Statistiques des prédictions
    print(f"\n📈 Statistiques de l'ensemble:")
    print(f"  - Moyenne: {final_predictions.mean():.6f}")
    print(f"  - Std: {final_predictions.std():.6f}")
    print(f"  - Min: {final_predictions.min():.6f}")
    print(f"  - Max: {final_predictions.max():.6f}")
    
    return final_predictions

# ========================================
# 6. FONCTION PRINCIPALE
# ========================================

def main():
    """Fonction principale d'exécution"""
    
    # 1. Chargement des données
    train_df, test_df = load_data()
    
    # 2. Feature Engineering
    train_enhanced = engineer_features(train_df, is_train=True)
    test_enhanced = engineer_features(test_df, is_train=False)
    
    # 3. Préparation des données
    feature_cols = [col for col in train_enhanced.columns 
                    if col not in ['Dates', 'ToPredict']]
    
    X_train = train_enhanced[feature_cols]
    y_train = train_enhanced['ToPredict']
    X_test = test_enhanced[feature_cols]
    
    print(f"\n📐 Dimensions finales:")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_test: {X_test.shape}")
    
    # 4. Normalisation avec RobustScaler
    print("\n🔧 Normalisation des données...")
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # 5. Obtenir les modèles
    models = get_models()
    
    # 6. Validation des modèles
    results_df = validate_models(X_train_scaled, y_train, models)
    
    # Afficher les résultats
    print("\n📊 Résultats de validation:")
    print(results_df.to_string(index=False))
    
    # 7. Créer l'ensemble pondéré avec les 3 meilleurs modèles
    final_predictions = create_weighted_ensemble(
        models, X_train_scaled, y_train, X_test_scaled, results_df, top_n=3
    )
    
    # 8. Créer le fichier de soumission
    submission = pd.DataFrame({
        'ID': test_df['Dates'].dt.strftime('%Y-%m-%d'),
        'ToPredict': final_predictions
    })
    
    submission.to_csv('submission_best.csv', index=False)
    print("\n✅ Fichier de soumission créé: submission_best.csv")
    
    # 9. Visualisation
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: Série temporelle des prédictions
    plt.subplot(1, 2, 1)
    plt.plot(test_df['Dates'], final_predictions, 'b-', alpha=0.7)
    plt.title('Prédictions sur l\'ensemble de test')
    plt.xlabel('Date')
    plt.ylabel('Valeur prédite')
    plt.xticks(rotation=45)
    
    # Subplot 2: Distribution des prédictions
    plt.subplot(1, 2, 2)
    plt.hist(final_predictions, bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.title('Distribution des prédictions')
    plt.xlabel('Valeur prédite')
    plt.ylabel('Fréquence')
    
    plt.tight_layout()
    plt.savefig('predictions_best.png', dpi=300)
    plt.show()
    
    # 10. Sauvegarder les résultats de validation
    results_df.to_csv('validation_results.csv', index=False)
    print("\n📊 Résultats de validation sauvegardés: validation_results.csv")
    
    print("\n✅ Analyse terminée avec succès!")
    print("\n🎯 Recommandations:")
    print("  - Les 3 meilleurs modèles ont été combinés avec des poids optimaux")
    print("  - RandomForest devrait avoir la plus forte pondération")
    print("  - L'approche évite le surapprentissage grâce à des paramètres conservateurs")

if __name__ == "__main__":
    main()