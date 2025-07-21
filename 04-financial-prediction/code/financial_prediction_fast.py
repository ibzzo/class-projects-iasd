#!/usr/bin/env python3
"""
Prédiction de Séries Temporelles Financières - Version Rapide et Optimisée
Basée sur la version simple avec quelques améliorations ciblées
Conçue pour une exécution rapide tout en maintenant de bonnes performances
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from typing import List, Tuple, Dict
import gc

warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import lightgbm as lgb
import xgboost as xgb

# Set random seeds
np.random.seed(42)

print("🚀 Financial Time Series Prediction - Version Rapide")
print("=" * 60)

# ========================================
# CHARGEMENT DES DONNÉES
# ========================================

def load_data(data_dir='data'):
    """Charge les données"""
    print("\n📊 Chargement des données...")
    
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')
    
    # Conversion des dates
    train_df['Dates'] = pd.to_datetime(train_df['Dates'])
    test_df['Dates'] = pd.to_datetime(test_df['Dates'])
    
    # Tri par date
    train_df = train_df.sort_values('Dates').reset_index(drop=True)
    test_df = test_df.sort_values('Dates').reset_index(drop=True)
    
    # ID pour la soumission
    if 'ID' not in test_df.columns:
        test_df['ID'] = test_df['Dates'].dt.strftime('%Y-%m-%d')
    
    print(f"✅ Données chargées: Train {train_df.shape}, Test {test_df.shape}")
    
    return train_df, test_df

# ========================================
# FEATURE ENGINEERING CIBLÉ
# ========================================

def create_date_features(df):
    """Features temporelles essentielles"""
    df = df.copy()
    
    # Features de base
    df['month'] = df['Dates'].dt.month
    df['dayofweek'] = df['Dates'].dt.dayofweek
    df['quarter'] = df['Dates'].dt.quarter
    df['dayofyear'] = df['Dates'].dt.dayofyear
    
    # Encodage cyclique
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    return df

def select_top_features(train_df, feature_cols, target_col='ToPredict', k=25):
    """Sélection rapide des meilleures features"""
    print(f"\n🔍 Sélection des {k} meilleures features...")
    
    # F-statistics uniquement pour la rapidité
    selector = SelectKBest(f_regression, k=k)
    selector.fit(train_df[feature_cols], train_df[target_col])
    
    # Récupérer les features sélectionnées
    selected_mask = selector.get_support()
    selected_features = [feat for feat, selected in zip(feature_cols, selected_mask) if selected]
    
    # Éviter Feature_38 si elle domine trop
    scores = pd.DataFrame({
        'feature': feature_cols,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    
    if scores.iloc[0]['score'] / scores.iloc[1]['score'] > 2:
        print(f"  ⚠️ {scores.iloc[0]['feature']} domine - exclusion")
        selected_features = [f for f in selected_features if f != scores.iloc[0]['feature']]
        # Ajouter la suivante
        next_feat = scores[~scores['feature'].isin(selected_features)].iloc[0]['feature']
        selected_features.append(next_feat)
    
    print(f"  ✅ {len(selected_features)} features sélectionnées")
    return selected_features[:k]

def create_simple_features(df, selected_features):
    """Features simples mais efficaces"""
    df = df.copy()
    
    # Rolling features pour top 5 seulement
    for col in selected_features[:5]:
        for window in [7, 30]:
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window, min_periods=int(window*0.5)).mean()
            df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window, min_periods=int(window*0.5)).std()
    
    # Lag features pour top 5
    for col in selected_features[:5]:
        for lag in [1, 7]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            df[f'{col}_diff_{lag}'] = df[col] - df[col].shift(lag)
    
    # Interactions simples (top 3 uniquement)
    for i in range(min(3, len(selected_features))):
        for j in range(i+1, min(3, len(selected_features))):
            col1, col2 = selected_features[i], selected_features[j]
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    
    return df

def feature_engineering_fast(train_df, test_df):
    """Pipeline de feature engineering rapide"""
    print("\n🔧 Feature Engineering...")
    
    # 1. Features temporelles
    train_df = create_date_features(train_df)
    test_df = create_date_features(test_df)
    
    # 2. Sélection des features
    feature_cols = [col for col in train_df.columns 
                   if col.startswith('Features_') and train_df[col].dtype in ['float64', 'int64']]
    selected_features = select_top_features(train_df, feature_cols, k=25)
    
    # 3. Créer des features dérivées
    train_df = create_simple_features(train_df, selected_features)
    test_df = create_simple_features(test_df, selected_features)
    
    print(f"✅ Feature engineering terminé: {train_df.shape[1]} colonnes")
    
    return train_df, test_df, selected_features

# ========================================
# MODÉLISATION RAPIDE
# ========================================

class FastEnsemble:
    """Ensemble rapide avec 3 modèles clés"""
    
    def __init__(self, n_splits=3):
        self.n_splits = n_splits
        self.models = {}
        self.weights = None
        
    def get_models(self):
        """3 modèles rapides et efficaces"""
        return {
            'lgb': lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=5,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbosity=-1,
                n_jobs=-1
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=5,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbosity=0,
                n_jobs=-1
            ),
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        }
    
    def fit(self, X, y):
        """Entraînement rapide avec validation temporelle"""
        print("\n🏗️ Entraînement des modèles...")
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        all_scores = {}
        oof_predictions = {}
        
        for name, model in self.get_models().items():
            print(f"\n  📊 {name}:", end=' ')
            
            oof_pred = np.zeros(len(y))
            scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Entraînement avec early stopping pour LGB et XGB
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
                else:
                    model.fit(X_train, y_train)
                
                # Prédictions
                val_pred = model.predict(X_val)
                oof_pred[val_idx] = val_pred
                
                score = np.sqrt(mean_squared_error(y_val, val_pred))
                scores.append(score)
            
            mean_score = np.mean(scores)
            print(f"RMSE: {mean_score:.6f}")
            
            oof_predictions[name] = oof_pred
            all_scores[name] = mean_score
            
            # Réentraîner sur toutes les données
            if name in ['lgb', 'xgb']:
                final_model = self.get_models()[name]
                if name == 'xgb':
                    final_model.set_params(early_stopping_rounds=None)
                final_model.fit(X, y)
                self.models[name] = final_model
            else:
                model.fit(X, y)
                self.models[name] = model
        
        # Optimiser les poids
        self._optimize_weights(oof_predictions, y)
        
        return all_scores
    
    def _optimize_weights(self, oof_predictions, y_true):
        """Optimisation simple des poids"""
        from scipy.optimize import minimize
        
        print("\n🎯 Optimisation des poids...")
        
        oof_array = np.column_stack([oof_predictions[name] for name in self.models.keys()])
        
        def objective(weights):
            weighted_pred = np.average(oof_array, axis=1, weights=weights)
            return np.sqrt(mean_squared_error(y_true, weighted_pred))
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0.1, 0.6) for _ in range(len(self.models))]
        x0 = np.ones(len(self.models)) / len(self.models)
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        self.weights = dict(zip(self.models.keys(), result.x))
        
        print("  Poids optimaux:", {k: f"{v:.3f}" for k, v in self.weights.items()})
        print(f"  Score final: {result.fun:.6f}")
    
    def predict(self, X):
        """Prédictions pondérées"""
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
            weights.append(self.weights[name])
        
        return np.average(np.array(predictions), axis=0, weights=weights)

# ========================================
# POST-PROCESSING SIMPLE
# ========================================

def simple_post_processing(predictions, train_target):
    """Post-processing minimal mais efficace"""
    print("\n🎯 Post-processing...")
    
    # 1. Clip aux percentiles du train
    lower = np.percentile(train_target, 1)
    upper = np.percentile(train_target, 99)
    predictions = np.clip(predictions, lower, upper)
    
    # 2. Ajuster légèrement la variance si nécessaire
    pred_std = predictions.std()
    train_std = train_target.std()
    if pred_std < train_std * 0.7:
        predictions = (predictions - predictions.mean()) * (train_std * 0.9 / pred_std) + train_target.mean()
    
    print(f"  Moyenne: {predictions.mean():.6f}, Std: {predictions.std():.6f}")
    
    return predictions

# ========================================
# PIPELINE PRINCIPAL
# ========================================

def main():
    """Pipeline principal rapide"""
    
    # 1. Charger les données
    train_df, test_df = load_data()
    
    # 2. Feature Engineering
    train_df, test_df, selected_features = feature_engineering_fast(train_df, test_df)
    
    # 3. Préparer les données
    exclude_cols = ['Dates', 'ToPredict', 'ID']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Gérer les NaN
    print("\n🧹 Gestion des valeurs manquantes...")
    for col in feature_cols:
        if train_df[col].isna().any() or test_df[col].isna().any():
            median_val = train_df[col].median()
            train_df[col] = train_df[col].fillna(median_val)
            test_df[col] = test_df[col].fillna(median_val)
    
    X_train = train_df[feature_cols].values
    y_train = train_df['ToPredict'].values
    X_test = test_df[feature_cols].values
    
    print(f"\n📐 Dimensions: X_train {X_train.shape}, X_test {X_test.shape}")
    
    # 4. Normalisation
    print("\n🔧 Normalisation...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Entraînement
    ensemble = FastEnsemble(n_splits=3)
    scores = ensemble.fit(X_train_scaled, y_train)
    
    # 6. Prédictions
    print("\n📈 Génération des prédictions...")
    predictions = ensemble.predict(X_test_scaled)
    
    # 7. Post-processing
    predictions = simple_post_processing(predictions, y_train)
    
    # 8. Sauvegarder
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'ToPredict': predictions
    })
    submission.to_csv('submission_fast.csv', index=False)
    print("\n✅ Fichier créé: submission_fast.csv")
    
    # 9. Visualisation simple
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(test_df['Dates'], predictions, alpha=0.7)
    plt.title('Prédictions')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.hist(predictions, bins=50, alpha=0.7, density=True, label='Pred')
    plt.hist(y_train, bins=50, alpha=0.7, density=True, label='Train')
    plt.legend()
    plt.title('Distributions')
    
    plt.tight_layout()
    plt.savefig('predictions_fast.png', dpi=300)
    print("📊 Graphique sauvegardé: predictions_fast.png")
    
    return scores, predictions

if __name__ == "__main__":
    scores, predictions = main()