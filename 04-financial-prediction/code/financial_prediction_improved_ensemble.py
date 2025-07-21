#!/usr/bin/env python3
"""
Prédiction de Séries Temporelles Financières - Version Ensemble Amélioré
Améliore l'équilibre entre les modèles et réduit la dominance de LightGBM
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
from sklearn.linear_model import Ridge, Lasso, HuberRegressor, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Set random seeds
np.random.seed(42)

print("🚀 Financial Time Series Prediction - Version Ensemble Amélioré")
print("=" * 60)

# ========================================
# FONCTIONS UTILITAIRES
# ========================================

def reduce_memory_usage(df):
    """Réduit l'usage mémoire du DataFrame"""
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
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
    
    train_df['Dates'] = pd.to_datetime(train_df['Dates'])
    test_df['Dates'] = pd.to_datetime(test_df['Dates'])
    
    train_df = train_df.sort_values('Dates').reset_index(drop=True)
    test_df = test_df.sort_values('Dates').reset_index(drop=True)
    
    if 'ID' not in test_df.columns:
        test_df['ID'] = test_df['Dates'].dt.strftime('%Y-%m-%d')
    
    print(f"✅ Données chargées:")
    print(f"   - Train: {train_df.shape}")
    print(f"   - Test: {test_df.shape}")
    print(f"   - Période train: {train_df['Dates'].min()} à {train_df['Dates'].max()}")
    print(f"   - Période test: {test_df['Dates'].min()} à {test_df['Dates'].max()}")
    
    return train_df, test_df

# ========================================
# FEATURE ENGINEERING ÉQUILIBRÉ
# ========================================

def create_date_features(df):
    """Crée des features temporelles basiques"""
    df = df.copy()
    
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
    
    return df

def select_important_features_balanced(train_df, feature_cols, target_col='ToPredict', k=25):
    """Sélection équilibrée des features sans dominance"""
    print(f"\n🔍 Sélection équilibrée des {k} features...")
    
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
    
    # Méthode 3: RandomForest importance
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
    
    # Score combiné équilibré
    all_scores['combined_score'] = (
        all_scores['f_score'] + 
        all_scores['mi_score'] + 
        all_scores['rf_importance']
    ) / 3
    
    # Appliquer une transformation logarithmique pour réduire les écarts
    all_scores['combined_score_log'] = np.log1p(all_scores['combined_score'] * 100)
    all_scores = all_scores.sort_values('combined_score_log', ascending=False)
    
    # Toujours exclure Features_38 si elle est dans le top
    selected_features = []
    for _, row in all_scores.iterrows():
        if row['feature'] != 'Features_38':
            selected_features.append(row['feature'])
        if len(selected_features) >= k:
            break
    
    print(f"\n📊 Top 10 features sélectionnées (équilibrées):")
    for i, feat in enumerate(selected_features[:10]):
        score = all_scores[all_scores['feature'] == feat]['combined_score_log'].values[0]
        print(f"   - {feat}: {score:.4f}")
    
    return selected_features[:k]

def create_minimal_rolling_features(df, feature_cols, windows=[7, 30]):
    """Rolling features minimales pour éviter l'overfitting"""
    df = df.copy()
    
    for col in feature_cols[:8]:  # Seulement top 8
        for window in windows:
            min_periods = int(window * 0.7)
            
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(
                window=window, min_periods=min_periods
            ).mean()
            
            df[f'{col}_roll_std_{window}'] = df[col].rolling(
                window=window, min_periods=min_periods
            ).std()
    
    return df

def create_minimal_lag_features(df, feature_cols, lags=[1, 7]):
    """Lag features minimales"""
    df = df.copy()
    
    for col in feature_cols[:8]:  # Seulement top 8
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            df[f'{col}_diff_{lag}'] = df[col] - df[col].shift(lag)
    
    return df

def feature_engineering_balanced(train_df, test_df):
    """Feature engineering équilibré pour éviter la dominance d'un modèle"""
    print("\n🔧 Feature Engineering Équilibré...")
    
    # 1. Features temporelles
    print("  - Features temporelles...")
    train_df = create_date_features(train_df)
    test_df = create_date_features(test_df)
    
    # 2. Identifier les colonnes numériques
    feature_cols = [col for col in train_df.columns 
                   if col.startswith('Features_') and train_df[col].dtype in ['float64', 'int64']]
    
    # 3. Sélection équilibrée des features
    selected_features = select_important_features_balanced(train_df, feature_cols, k=25)
    
    # 4. Features dérivées minimales
    print("\n  - Création de features dérivées minimales...")
    train_df = create_minimal_rolling_features(train_df, selected_features)
    test_df = create_minimal_rolling_features(test_df, selected_features)
    
    train_df = create_minimal_lag_features(train_df, selected_features)
    test_df = create_minimal_lag_features(test_df, selected_features)
    
    # 5. Interactions très limitées (top 4 seulement)
    print("  - Création d'interactions limitées...")
    top_4_features = selected_features[:4]
    for i in range(len(top_4_features)):
        for j in range(i+1, len(top_4_features)):
            col1, col2 = top_4_features[i], top_4_features[j]
            train_df[f'{col1}_x_{col2}'] = train_df[col1] * train_df[col2]
            test_df[f'{col1}_x_{col2}'] = test_df[col1] * test_df[col2]
    
    print(f"\n✅ Feature engineering terminé:")
    print(f"   - Train: {train_df.shape}")
    print(f"   - Test: {test_df.shape}")
    
    return train_df, test_df

# ========================================
# MODÉLISATION AVEC ENSEMBLE ÉQUILIBRÉ
# ========================================

class BalancedTimeSeriesEnsemble:
    """Ensemble équilibré pour éviter la dominance d'un modèle"""
    
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.models = {}
        self.oof_predictions = None
        self.feature_importance = None
        
    def get_models(self):
        """Modèles avec hyperparamètres optimisés pour équilibrer les performances"""
        return {
            # LightGBM avec régularisation plus forte
            'lgb': lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                n_estimators=800,
                learning_rate=0.02,
                num_leaves=20,  # Réduit pour moins d'overfitting
                max_depth=4,    # Réduit
                min_child_samples=30,  # Augmenté
                subsample=0.7,
                subsample_freq=1,
                colsample_bytree=0.7,
                reg_alpha=0.2,  # Plus de régularisation
                reg_lambda=0.2,
                random_state=42,
                verbose=-1
            ),
            # XGBoost optimisé
            'xgb': xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=800,
                learning_rate=0.02,
                max_depth=4,
                min_child_weight=5,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.15,
                reg_lambda=0.15,
                gamma=0.1,
                random_state=42,
                verbosity=0
            ),
            # CatBoost avec paramètres ajustés
            'cat': CatBoostRegressor(
                iterations=800,
                learning_rate=0.03,
                depth=5,
                l2_leaf_reg=5,
                min_data_in_leaf=25,
                random_strength=0.5,
                bagging_temperature=0.3,
                od_type='Iter',
                od_wait=50,
                random_state=42,
                verbose=False
            ),
            # RandomForest simplifié
            'rf': RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1
            ),
            # GradientBoosting ajusté
            'gb': GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.02,
                max_depth=4,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.7,
                max_features='sqrt',
                random_state=42
            ),
            # Modèles linéaires pour diversité
            'ridge': Ridge(
                alpha=1.0,
                random_state=42
            ),
            'elastic': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=42
            )
        }
    
    def fit(self, X, y, feature_names=None):
        """Entraîne les modèles avec stratégie d'ensemble équilibrée"""
        print("\n🏗️ Entraînement avec ensemble équilibré...")
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        self.oof_predictions = {}
        all_scores = {}
        
        for name, model in self.get_models().items():
            print(f"\n  📊 Modèle: {name}")
            
            oof_pred = np.zeros(len(y))
            scores = []
            
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
                
                val_pred = model.predict(X_val)
                oof_pred[val_idx] = val_pred
                
                score = np.sqrt(mean_squared_error(y_val, val_pred))
                scores.append(score)
                print(f"RMSE: {score:.6f}")
            
            self.oof_predictions[name] = oof_pred
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            all_scores[name] = {'mean': mean_score, 'std': std_score}
            
            print(f"    Score moyen: {mean_score:.6f} (+/- {std_score:.6f})")
            
            # Réentraîner sur toutes les données
            print(f"    Entraînement final...")
            if name in ['lgb', 'xgb', 'cat']:
                # Créer un nouveau modèle sans early stopping
                final_model = self.get_models()[name]
                if name == 'lgb':
                    final_model.fit(X, y, callbacks=[lgb.log_evaluation(0)])
                elif name == 'xgb':
                    final_model.set_params(early_stopping_rounds=None)
                    final_model.fit(X, y, verbose=False)
                else:
                    final_model.fit(X, y, verbose=False)
                self.models[name] = final_model
            else:
                model.fit(X, y)
                self.models[name] = model
        
        # Optimiser les poids avec contraintes
        self._optimize_weights_balanced(y)
        
        return all_scores
    
    def _optimize_weights_balanced(self, y_true):
        """Optimise les poids avec contraintes pour équilibrer l'ensemble"""
        from scipy.optimize import minimize
        
        print("\n🎯 Optimisation des poids avec contraintes d'équilibre...")
        
        oof_array = np.column_stack([self.oof_predictions[name] for name in self.models.keys()])
        
        def objective(weights):
            weighted_pred = np.average(oof_array, axis=1, weights=weights)
            return np.sqrt(mean_squared_error(y_true, weighted_pred))
        
        # Contraintes pour éviter la dominance
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Somme = 1
            {'type': 'ineq', 'fun': lambda w: 0.5 - np.max(w)}  # Aucun poids > 0.5
        ]
        
        # Bornes: minimum 5% pour chaque modèle
        bounds = [(0.05, 0.5) for _ in range(len(self.models))]
        
        # Point de départ uniforme
        x0 = np.ones(len(self.models)) / len(self.models)
        
        # Optimisation
        result = minimize(
            objective, x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        # Si l'optimisation échoue, utiliser des poids basés sur les performances
        if not result.success:
            print("  ⚠️ Optimisation avec contraintes échouée, utilisation de poids basés sur les performances")
            # Calculer les poids inversement proportionnels aux erreurs
            model_errors = [np.sqrt(mean_squared_error(y_true, self.oof_predictions[name])) 
                          for name in self.models.keys()]
            inverse_errors = [1/e for e in model_errors]
            total = sum(inverse_errors)
            weights = [w/total for w in inverse_errors]
            
            # Appliquer la contrainte de max 0.5
            weights = np.array(weights)
            while np.max(weights) > 0.5:
                max_idx = np.argmax(weights)
                excess = weights[max_idx] - 0.5
                weights[max_idx] = 0.5
                # Redistribuer l'excès
                other_indices = [i for i in range(len(weights)) if i != max_idx]
                for idx in other_indices:
                    weights[idx] += excess / len(other_indices)
            
            self.weights = dict(zip(self.models.keys(), weights))
        else:
            self.weights = dict(zip(self.models.keys(), result.x))
        
        print("\n📊 Poids équilibrés:")
        for name, weight in self.weights.items():
            print(f"   - {name}: {weight:.3f}")
        
        # Calculer le score final
        weighted_pred = np.average(oof_array, axis=1, 
                                 weights=[self.weights[name] for name in self.models.keys()])
        final_score = np.sqrt(mean_squared_error(y_true, weighted_pred))
        print(f"\n   Score ensemble équilibré (RMSE): {final_score:.6f}")
    
    def predict(self, X):
        """Prédictions avec l'ensemble équilibré"""
        predictions = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        pred_array = np.column_stack([predictions[name] for name in self.models.keys()])
        weights_array = np.array([self.weights[name] for name in self.models.keys()])
        
        return np.average(pred_array, axis=1, weights=weights_array)

# ========================================
# POST-PROCESSING
# ========================================

def post_process_predictions(predictions, train_target, test_size):
    """Post-processing conservateur"""
    print("\n🎯 Post-processing des prédictions...")
    
    # 1. Gérer les valeurs négatives
    min_train = train_target.min()
    if min_train >= 0 and predictions.min() < 0:
        print(f"  - Correction des valeurs négatives ({np.sum(predictions < 0)} valeurs)")
        predictions = np.maximum(predictions, 0)
    
    # 2. Gérer les valeurs extrêmes
    lower_bound = np.percentile(train_target, 1)
    upper_bound = np.percentile(train_target, 99)
    
    n_clipped = np.sum((predictions < lower_bound) | (predictions > upper_bound))
    if n_clipped > 0:
        print(f"  - Clipping de {n_clipped} valeurs extrêmes")
        predictions = np.clip(predictions, lower_bound, upper_bound)
    
    # 3. Ajustement léger de la distribution
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    train_mean = train_target.mean()
    train_std = train_target.std()
    
    # Ajustement très conservateur
    if pred_std < 0.7 * train_std:
        print(f"  - Ajustement léger de la variance")
        predictions = (predictions - pred_mean) * (train_std * 0.85 / pred_std) + train_mean
    
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
    """Pipeline principal avec ensemble équilibré"""
    
    # 1. Charger les données
    train_df, test_df = load_data()
    
    # 2. Feature Engineering équilibré
    train_df, test_df = feature_engineering_balanced(train_df, test_df)
    
    # 3. Préparer les données
    exclude_cols = ['Dates', 'ToPredict', 'ID']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Gérer les valeurs manquantes
    print("\n🧹 Gestion des valeurs manquantes...")
    for col in feature_cols:
        if 'lag' in col or 'roll' in col or 'diff' in col:
            train_df[col] = train_df[col].fillna(method='ffill').fillna(0)
            test_df[col] = test_df[col].fillna(method='ffill').fillna(0)
        else:
            median_val = train_df[col].median()
            train_df[col] = train_df[col].fillna(median_val)
            test_df[col] = test_df[col].fillna(median_val)
    
    X_train = train_df[feature_cols].values
    y_train = train_df['ToPredict'].values
    X_test = test_df[feature_cols].values
    
    print(f"\n📐 Dimensions finales:")
    print(f"   - X_train: {X_train.shape}")
    print(f"   - X_test: {X_test.shape}")
    
    # 4. Normalisation
    print("\n🔧 Normalisation des données...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Entraînement avec ensemble équilibré
    ensemble = BalancedTimeSeriesEnsemble(n_splits=5)
    scores = ensemble.fit(X_train_scaled, y_train, feature_names=feature_cols)
    
    # 6. Prédictions
    print("\n📈 Génération des prédictions...")
    predictions = ensemble.predict(X_test_scaled)
    
    # 7. Post-processing
    predictions = post_process_predictions(predictions, y_train, len(X_test))
    
    # 8. Sauvegarder les résultats
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'ToPredict': predictions
    })
    submission.to_csv('submission_improved_ensemble.csv', index=False)
    print("\n✅ Fichier de soumission créé: submission_improved_ensemble.csv")
    
    # 9. Visualisation
    plt.figure(figsize=(15, 10))
    
    # Comparaison des distributions
    plt.subplot(2, 2, 1)
    plt.hist(predictions, bins=50, alpha=0.7, label='Predictions', density=True, color='blue')
    plt.hist(y_train, bins=50, alpha=0.7, label='Train', density=True, color='orange')
    plt.xlabel('Target Value')
    plt.ylabel('Density')
    plt.title('Distribution Comparison')
    plt.legend()
    
    # Prédictions dans le temps
    plt.subplot(2, 2, 2)
    plt.plot(test_df['Dates'], predictions, alpha=0.7, color='blue')
    plt.xlabel('Date')
    plt.ylabel('Target')
    plt.title('Predictions Over Time')
    plt.xticks(rotation=45)
    
    # Comparaison des modèles
    plt.subplot(2, 2, 3)
    model_names = list(ensemble.weights.keys())
    model_weights = list(ensemble.weights.values())
    plt.bar(model_names, model_weights)
    plt.xlabel('Model')
    plt.ylabel('Weight')
    plt.title('Model Weights in Ensemble')
    plt.xticks(rotation=45)
    
    # Scores des modèles
    plt.subplot(2, 2, 4)
    model_means = [scores[name]['mean'] for name in model_names if name in scores]
    model_stds = [scores[name]['std'] for name in model_names if name in scores]
    x_pos = np.arange(len(model_names))
    plt.bar(x_pos, model_means, yerr=model_stds, capsize=5)
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.title('Model Performance (CV)')
    plt.xticks(x_pos, model_names, rotation=45)
    
    plt.tight_layout()
    plt.savefig('predictions_improved_ensemble_analysis.png', dpi=300, bbox_inches='tight')
    print("\n📊 Graphiques sauvegardés: predictions_improved_ensemble_analysis.png")
    
    return scores, predictions

if __name__ == "__main__":
    scores, predictions = main()