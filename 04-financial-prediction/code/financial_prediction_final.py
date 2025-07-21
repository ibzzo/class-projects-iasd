#!/usr/bin/env python3
"""
Prédiction de Séries Temporelles Financières - Version Finale Optimisée
Combine les meilleures pratiques des versions précédentes
Focus sur la simplicité, la robustesse et la performance
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
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Set random seeds
np.random.seed(42)

print("🚀 Financial Time Series Prediction - Version Finale")
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
    
    # Tri par date - CRUCIAL pour time series
    train_df = train_df.sort_values('Dates').reset_index(drop=True)
    test_df = test_df.sort_values('Dates').reset_index(drop=True)
    
    # ID pour la soumission
    if 'ID' not in test_df.columns:
        test_df['ID'] = test_df['Dates'].dt.strftime('%Y-%m-%d')
    
    print(f"✅ Données chargées:")
    print(f"   - Train: {train_df.shape}")
    print(f"   - Test: {test_df.shape}")
    print(f"   - Période train: {train_df['Dates'].min().date()} à {train_df['Dates'].max().date()}")
    print(f"   - Période test: {test_df['Dates'].min().date()} à {test_df['Dates'].max().date()}")
    
    # Stats de la target
    target_stats = {
        'mean': train_df['ToPredict'].mean(),
        'std': train_df['ToPredict'].std(),
        'min': train_df['ToPredict'].min(),
        'max': train_df['ToPredict'].max()
    }
    
    return train_df, test_df, target_stats

# ========================================
# FEATURE ENGINEERING ÉQUILIBRÉ
# ========================================

def create_temporal_features(df):
    """Features temporelles avec encodage cyclique"""
    df = df.copy()
    
    # Features de base
    df['year'] = df['Dates'].dt.year
    df['month'] = df['Dates'].dt.month
    df['day'] = df['Dates'].dt.day
    df['dayofweek'] = df['Dates'].dt.dayofweek
    df['quarter'] = df['Dates'].dt.quarter
    df['dayofyear'] = df['Dates'].dt.dayofyear
    df['weekofyear'] = df['Dates'].dt.isocalendar().week
    
    # Encodage cyclique pour capturer la périodicité
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Indicateurs booléens
    df['is_month_start'] = (df['day'] <= 5).astype(int)
    df['is_month_end'] = (df['day'] >= 25).astype(int)
    
    return df

def select_balanced_features(train_df, feature_cols, target_col='ToPredict', k=30):
    """Sélection équilibrée des features pour éviter la domination"""
    print(f"\n🔍 Sélection équilibrée de {k} features...")
    
    # Méthode 1: F-statistics
    selector_f = SelectKBest(f_regression, k=min(k*2, len(feature_cols)))
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
    
    # Méthode 3: RandomForest (rapide)
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(train_df[feature_cols], train_df[target_col])
    scores_rf = pd.DataFrame({
        'feature': feature_cols,
        'rf_importance': rf.feature_importances_
    }).sort_values('rf_importance', ascending=False)
    
    # Combiner les scores
    all_scores = scores_f.merge(scores_mi, on='feature').merge(scores_rf, on='feature')
    
    # Normaliser
    for col in ['f_score', 'mi_score', 'rf_importance']:
        min_val = all_scores[col].min()
        max_val = all_scores[col].max()
        if max_val > min_val:
            all_scores[col] = (all_scores[col] - min_val) / (max_val - min_val)
    
    # Score combiné équilibré
    all_scores['combined_score'] = (
        0.40 * all_scores['f_score'] + 
        0.30 * all_scores['mi_score'] + 
        0.30 * all_scores['rf_importance']
    )
    all_scores = all_scores.sort_values('combined_score', ascending=False)
    
    # Vérifier la domination et diversifier
    selected_features = []
    feature_prefixes = {}
    
    for _, row in all_scores.iterrows():
        feature = row['feature']
        prefix = feature.split('_')[0]
        
        # Éviter la feature dominante
        if len(selected_features) == 0 and row['combined_score'] > 0.9:
            print(f"  ⚠️ {feature} exclue (score trop élevé: {row['combined_score']:.3f})")
            continue
        
        # Limiter les features du même préfixe
        if prefix not in feature_prefixes:
            feature_prefixes[prefix] = 0
        
        if feature_prefixes[prefix] < 3:  # Max 3 par préfixe
            selected_features.append(feature)
            feature_prefixes[prefix] += 1
        
        if len(selected_features) >= k:
            break
    
    print(f"  ✅ {len(selected_features)} features sélectionnées (diversité: {len(feature_prefixes)} groupes)")
    
    return selected_features

def create_engineered_features(df, selected_features):
    """Création de features dérivées équilibrées"""
    df = df.copy()
    
    # 1. Rolling features (top 8 features, fenêtres modérées)
    for i, col in enumerate(selected_features[:8]):
        for window in [7, 21]:  # Fenêtres plus petites
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window, min_periods=int(window*0.6)).mean()
            df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window, min_periods=int(window*0.6)).std()
            
            # Ratio avec la moyenne mobile
            df[f'{col}_roll_ratio_{window}'] = df[col] / (df[f'{col}_roll_mean_{window}'] + 1e-8)
    
    # 2. Lag features (top 8 features)
    for col in selected_features[:8]:
        for lag in [1, 7, 21]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            df[f'{col}_diff_{lag}'] = df[col] - df[col].shift(lag)
    
    # 3. Interactions (top 5 uniquement)
    for i in range(min(5, len(selected_features))):
        for j in range(i+1, min(5, len(selected_features))):
            col1, col2 = selected_features[i], selected_features[j]
            # Produit
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            # Ratio
            df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
    
    # 4. Features statistiques globales (top 5)
    for col in selected_features[:5]:
        # Expanding statistics
        df[f'{col}_expanding_mean'] = df[col].expanding(min_periods=30).mean()
        df[f'{col}_expanding_std'] = df[col].expanding(min_periods=30).std()
    
    return df

def feature_engineering_pipeline(train_df, test_df):
    """Pipeline complet de feature engineering"""
    print("\n🔧 Feature Engineering...")
    
    # 1. Features temporelles
    print("  - Création des features temporelles...")
    train_df = create_temporal_features(train_df)
    test_df = create_temporal_features(test_df)
    
    # 2. Identification des features numériques
    feature_cols = [col for col in train_df.columns 
                   if col.startswith('Features_') and train_df[col].dtype in ['float64', 'int64']]
    
    print(f"  - {len(feature_cols)} features numériques trouvées")
    
    # 3. Sélection équilibrée
    selected_features = select_balanced_features(train_df, feature_cols, k=30)
    
    # 4. Création de features dérivées
    print("  - Création de features dérivées...")
    train_df = create_engineered_features(train_df, selected_features)
    test_df = create_engineered_features(test_df, selected_features)
    
    print(f"✅ Feature engineering terminé: {train_df.shape[1]} colonnes")
    
    return train_df, test_df

# ========================================
# ENSEMBLE DE MODÈLES OPTIMISÉ
# ========================================

class OptimizedEnsemble:
    """Ensemble optimisé avec validation temporelle stricte"""
    
    def __init__(self, n_splits=4):
        self.n_splits = n_splits
        self.models = {}
        self.oof_predictions = None
        self.weights = None
        
    def get_models(self):
        """Modèles optimisés pour la compétition"""
        return {
            'lgb': lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                n_estimators=800,
                learning_rate=0.015,
                num_leaves=31,
                max_depth=5,
                min_child_samples=20,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbosity=-1
            ),
            'xgb': xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=800,
                learning_rate=0.015,
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
                iterations=800,
                learning_rate=0.025,
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
                learning_rate=0.015,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42
            )
        }
    
    def fit(self, X, y):
        """Entraînement avec validation temporelle"""
        print("\n🏗️ Entraînement de l'ensemble...")
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        self.oof_predictions = {}
        all_scores = {}
        
        for name, model in self.get_models().items():
            print(f"\n  📊 {name}:")
            
            oof_pred = np.zeros(len(y))
            scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                print(f"    Fold {fold_idx + 1}/{self.n_splits}", end=' ')
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Entraînement avec callbacks appropriés
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
                
                # Score
                score = np.sqrt(mean_squared_error(y_val, val_pred))
                scores.append(score)
                print(f"RMSE: {score:.6f}")
            
            # Stocker les résultats
            self.oof_predictions[name] = oof_pred
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            all_scores[name] = {'mean': mean_score, 'std': std_score}
            
            print(f"    Moyenne: {mean_score:.6f} (+/- {std_score:.6f})")
            
            # Réentraîner sur l'ensemble complet
            print(f"    Entraînement final...")
            if name in ['lgb', 'xgb', 'cat']:
                final_model = self.get_models()[name]
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
        
        # Optimiser les poids
        self._optimize_weights(y)
        
        return all_scores
    
    def _optimize_weights(self, y_true):
        """Optimisation des poids avec contraintes"""
        from scipy.optimize import minimize
        
        print("\n🎯 Optimisation des poids de l'ensemble...")
        
        oof_array = np.column_stack([self.oof_predictions[name] for name in self.models.keys()])
        
        def objective(weights):
            weighted_pred = np.average(oof_array, axis=1, weights=weights)
            return np.sqrt(mean_squared_error(y_true, weighted_pred))
        
        # Contraintes et bornes
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0.05, 0.5) for _ in range(len(self.models))]
        
        # Point de départ
        x0 = np.ones(len(self.models)) / len(self.models)
        
        # Optimisation
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        self.weights = dict(zip(self.models.keys(), result.x))
        
        print("\n  Poids optimaux:")
        for name, weight in self.weights.items():
            print(f"    - {name}: {weight:.3f}")
        
        print(f"\n  Score optimal (RMSE): {result.fun:.6f}")
    
    def predict(self, X):
        """Prédictions pondérées de l'ensemble"""
        predictions = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # Moyenne pondérée
        pred_array = np.column_stack([predictions[name] for name in self.models.keys()])
        weights_array = np.array([self.weights[name] for name in self.models.keys()])
        
        return np.average(pred_array, axis=1, weights=weights_array)

# ========================================
# POST-PROCESSING INTELLIGENT
# ========================================

def intelligent_post_processing(predictions, train_target, target_stats):
    """Post-processing adaptatif basé sur les statistiques"""
    print("\n🎯 Post-processing intelligent...")
    
    # 1. Clipping aux percentiles
    lower_bound = np.percentile(train_target, 0.5)
    upper_bound = np.percentile(train_target, 99.5)
    
    n_clipped = np.sum((predictions < lower_bound) | (predictions > upper_bound))
    if n_clipped > 0:
        print(f"  - Clipping de {n_clipped} valeurs extrêmes")
        predictions = np.clip(predictions, lower_bound, upper_bound)
    
    # 2. Ajustement de la distribution
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    
    # Ajuster légèrement vers les stats du train
    target_mean = target_stats['mean']
    target_std = target_stats['std']
    
    # Blend: 80% predictions, 20% target stats
    adjusted_mean = 0.8 * pred_mean + 0.2 * target_mean
    adjusted_std = 0.9 * pred_std + 0.1 * target_std
    
    if abs(pred_std - target_std) / target_std > 0.3:
        print(f"  - Ajustement de la distribution (std: {pred_std:.4f} -> {adjusted_std:.4f})")
        predictions = (predictions - pred_mean) * (adjusted_std / pred_std) + adjusted_mean
    
    # 3. Lissage léger si beaucoup de prédictions
    if len(predictions) > 500:
        # Lissage très léger pour réduire le bruit
        smoothed = pd.Series(predictions).rolling(window=3, center=True, min_periods=1).mean().values
        predictions = 0.9 * predictions + 0.1 * smoothed
    
    print(f"\n  Statistiques finales:")
    print(f"    - Moyenne: {predictions.mean():.6f}")
    print(f"    - Std: {predictions.std():.6f}")
    print(f"    - Min: {predictions.min():.6f}")
    print(f"    - Max: {predictions.max():.6f}")
    
    return predictions

# ========================================
# PIPELINE PRINCIPAL
# ========================================

def main():
    """Pipeline principal optimisé"""
    
    # 1. Chargement des données
    train_df, test_df, target_stats = load_data()
    
    # 2. Feature Engineering
    train_df, test_df = feature_engineering_pipeline(train_df, test_df)
    
    # 3. Préparation des données
    exclude_cols = ['Dates', 'ToPredict', 'ID']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Gestion des valeurs manquantes
    print("\n🧹 Gestion des valeurs manquantes...")
    for col in feature_cols:
        if train_df[col].isna().any() or test_df[col].isna().any():
            if 'lag' in col or 'roll' in col or 'diff' in col or 'expanding' in col:
                # Pour les features temporelles
                train_median = train_df[col].median()
                if pd.isna(train_median):
                    train_median = 0
                train_df[col] = train_df[col].fillna(train_median)
                test_df[col] = test_df[col].fillna(train_median)
            else:
                # Pour les autres
                median_val = train_df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                train_df[col] = train_df[col].fillna(median_val)
                test_df[col] = test_df[col].fillna(median_val)
    
    # Conversion en arrays
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
    
    # 5. Entraînement de l'ensemble
    ensemble = OptimizedEnsemble(n_splits=4)
    scores = ensemble.fit(X_train_scaled, y_train)
    
    # 6. Prédictions
    print("\n📈 Génération des prédictions...")
    predictions = ensemble.predict(X_test_scaled)
    
    # 7. Post-processing
    predictions = intelligent_post_processing(predictions, y_train, target_stats)
    
    # 8. Sauvegarde
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'ToPredict': predictions
    })
    submission.to_csv('submission_final.csv', index=False)
    print("\n✅ Fichier de soumission créé: submission_final.csv")
    
    # 9. Visualisation complète
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Série temporelle
    ax = axes[0, 0]
    ax.plot(test_df['Dates'], predictions, 'b-', alpha=0.7, linewidth=1)
    ax.set_title('Prédictions sur la période de test')
    ax.set_xlabel('Date')
    ax.set_ylabel('Valeur prédite')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Distribution
    ax = axes[0, 1]
    ax.hist(predictions, bins=50, alpha=0.7, density=True, color='blue', edgecolor='black', label='Prédictions')
    ax.hist(y_train, bins=50, alpha=0.7, density=True, color='orange', edgecolor='black', label='Train')
    ax.set_title('Comparaison des distributions')
    ax.set_xlabel('Valeur')
    ax.set_ylabel('Densité')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Q-Q plot
    ax = axes[1, 0]
    from scipy import stats
    stats.probplot(predictions, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (normalité)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Scores des modèles
    ax = axes[1, 1]
    model_names = list(scores.keys())
    model_scores = [scores[name]['mean'] for name in model_names]
    model_stds = [scores[name]['std'] for name in model_names]
    
    x_pos = np.arange(len(model_names))
    ax.bar(x_pos, model_scores, yerr=model_stds, capsize=5, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names)
    ax.set_title('Performance des modèles (RMSE)')
    ax.set_ylabel('RMSE')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('predictions_final_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Graphiques sauvegardés: predictions_final_analysis.png")
    
    # 10. Résumé final
    print("\n" + "="*60)
    print("📊 RÉSUMÉ FINAL")
    print("="*60)
    print(f"Meilleur modèle individuel: {min(scores.items(), key=lambda x: x[1]['mean'])[0]}")
    print(f"Score ensemble optimisé: ~{np.mean([s['mean'] for s in scores.values()]):.6f}")
    print(f"Nombre de features utilisées: {X_train.shape[1]}")
    print(f"Ratio variance (pred/train): {predictions.std() / y_train.std():.4f}")
    print("="*60)
    
    return scores, predictions

if __name__ == "__main__":
    scores, predictions = main()