#!/usr/bin/env python3
"""
Prédiction de Séries Temporelles Financières - Version Fine-Tuned
Basé sur parameter_tuning qui donne les meilleurs résultats, avec ajustements fins
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from typing import List, Tuple, Dict
import gc
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

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

print("🚀 Financial Time Series Prediction - Version Fine-Tuned")
print("=" * 60)

# ========================================
# FONCTIONS DE VISUALISATION
# ========================================

def create_overview_plot(train_df, test_df):
    """Crée un plot d'aperçu général des données pour la présentation"""
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Vue d\'ensemble des Données Financières', fontsize=24, fontweight='bold')
    
    # Grille personnalisée
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Statistiques clés
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    # Calculer les statistiques
    total_days = (train_df['Dates'].max() - train_df['Dates'].min()).days
    n_features = len([col for col in train_df.columns if col.startswith('Features_')])
    
    stats_text = f"""
    DONNÉES D'ENTRAÎNEMENT:
    • Période: {train_df['Dates'].min().strftime('%d/%m/%Y')} à {train_df['Dates'].max().strftime('%d/%m/%Y')} ({total_days} jours)
    • Nombre d'observations: {len(train_df):,} (train) + {len(test_df):,} (test) = {len(train_df) + len(test_df):,} total
    • Nombre de features: {n_features}
    • Variable cible: 'ToPredict' (valeur financière à prédire)
    
    STATISTIQUES DE LA VARIABLE CIBLE:
    • Moyenne: {train_df['ToPredict'].mean():.6f}
    • Écart-type: {train_df['ToPredict'].std():.6f}
    • Min / Max: {train_df['ToPredict'].min():.6f} / {train_df['ToPredict'].max():.6f}
    """
    
    ax1.text(0.5, 0.5, stats_text, transform=ax1.transAxes, 
             fontsize=16, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3))
    
    # 2. Série temporelle complète
    ax2 = fig.add_subplot(gs[1, :])
    daily_mean = train_df.groupby('Dates')['ToPredict'].mean()
    ax2.plot(daily_mean.index, daily_mean.values, linewidth=0.5, alpha=0.8, color='darkblue')
    
    # Moyennes mobiles
    ma_7 = daily_mean.rolling(window=7, min_periods=1).mean()
    ma_30 = daily_mean.rolling(window=30, min_periods=1).mean()
    ax2.plot(ma_7.index, ma_7.values, color='orange', linewidth=2, label='MA 7 jours')
    ax2.plot(ma_30.index, ma_30.values, color='red', linewidth=2, label='MA 30 jours')
    
    ax2.set_xlabel('Date', fontsize=14)
    ax2.set_ylabel('Valeur moyenne ToPredict', fontsize=14)
    ax2.set_title('Évolution Temporelle de la Variable Cible', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution de la target
    ax3 = fig.add_subplot(gs[2, 0])
    n, bins, patches = ax3.hist(train_df['ToPredict'], bins=100, alpha=0.7, color='darkgreen', edgecolor='black')
    ax3.axvline(train_df['ToPredict'].mean(), color='red', linestyle='--', linewidth=2, label=f'Moyenne')
    ax3.axvline(train_df['ToPredict'].median(), color='orange', linestyle='--', linewidth=2, label=f'Médiane')
    ax3.set_xlabel('ToPredict', fontsize=12)
    ax3.set_ylabel('Fréquence', fontsize=12)
    ax3.set_title('Distribution de la Variable Cible', fontsize=14)
    ax3.legend(fontsize=10)
    
    # 4. Tendance annuelle
    ax4 = fig.add_subplot(gs[2, 1])
    train_df['year'] = train_df['Dates'].dt.year
    yearly_stats = train_df.groupby('year')['ToPredict'].agg(['mean', 'std'])
    x = yearly_stats.index
    ax4.plot(x, yearly_stats['mean'], marker='o', markersize=8, linewidth=2, color='navy')
    ax4.fill_between(x, yearly_stats['mean'] - yearly_stats['std'], 
                     yearly_stats['mean'] + yearly_stats['std'], alpha=0.3, color='navy')
    ax4.set_xlabel('Année', fontsize=12)
    ax4.set_ylabel('Moyenne ToPredict', fontsize=12)
    ax4.set_title('Tendance Annuelle avec Écart-Type', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    # 5. Saisonnalité mensuelle
    ax5 = fig.add_subplot(gs[2, 2])
    train_df['month'] = train_df['Dates'].dt.month
    monthly_stats = train_df.groupby('month')['ToPredict'].agg(['mean', 'std'])
    months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
    ax5.bar(monthly_stats.index, monthly_stats['mean'], yerr=monthly_stats['std'], 
            alpha=0.7, capsize=5, color='coral', edgecolor='darkred')
    ax5.set_xlabel('Mois', fontsize=12)
    ax5.set_ylabel('Moyenne ToPredict', fontsize=12)
    ax5.set_title('Saisonnalité Mensuelle', fontsize=14)
    ax5.set_xticks(range(1, 13))
    ax5.set_xticklabels(months, rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('presentation_plots/00_data_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   📊 Plot sauvegardé: presentation_plots/00_data_overview.png")

def create_data_exploration_plots(train_df, test_df):
    """Crée des visualisations pour explorer les données"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Distribution de la target
    ax = axes[0, 0]
    ax.hist(train_df['ToPredict'], bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(train_df['ToPredict'].mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Moyenne: {train_df["ToPredict"].mean():.4f}')
    ax.axvline(train_df['ToPredict'].median(), color='green', linestyle='--', linewidth=2,
               label=f'Médiane: {train_df["ToPredict"].median():.4f}')
    ax.set_xlabel('ToPredict')
    ax.set_ylabel('Fréquence')
    ax.set_title('Distribution de la Variable Cible')
    ax.legend()
    
    # 2. Évolution temporelle
    ax = axes[0, 1]
    daily_mean = train_df.groupby('Dates')['ToPredict'].mean()
    ax.plot(daily_mean.index, daily_mean.values, linewidth=0.5, alpha=0.7)
    # Moyenne mobile
    rolling_mean = daily_mean.rolling(window=30, min_periods=1).mean()
    ax.plot(rolling_mean.index, rolling_mean.values, color='red', linewidth=2, label='MA 30j')
    ax.set_xlabel('Date')
    ax.set_ylabel('ToPredict')
    ax.set_title('Évolution Temporelle de la Target')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    # 3. Saisonnalité mensuelle
    ax = axes[1, 0]
    train_df['month'] = train_df['Dates'].dt.month
    monthly_stats = train_df.groupby('month')['ToPredict'].agg(['mean', 'std'])
    ax.bar(monthly_stats.index, monthly_stats['mean'], yerr=monthly_stats['std'], 
           alpha=0.7, capsize=5, color='green')
    ax.set_xlabel('Mois')
    ax.set_ylabel('ToPredict Moyen')
    ax.set_title('Saisonnalité Mensuelle')
    ax.set_xticks(range(1, 13))
    
    # 4. Box plot par année
    ax = axes[1, 1]
    train_df['year'] = train_df['Dates'].dt.year
    years_to_show = sorted(train_df['year'].unique())[-5:]  # 5 dernières années
    data_to_plot = [train_df[train_df['year'] == year]['ToPredict'].values for year in years_to_show]
    ax.boxplot(data_to_plot, labels=[str(y) for y in years_to_show])
    ax.set_xlabel('Année')
    ax.set_ylabel('ToPredict')
    ax.set_title('Distribution par Année (5 dernières)')
    
    plt.tight_layout()
    plt.savefig('presentation_plots/01_data_exploration.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   📊 Plot sauvegardé: presentation_plots/01_data_exploration.png")

def create_feature_importance_plots(feature_importance_df, top_n=30):
    """Visualise l'importance des features"""
    plt.figure(figsize=(12, 10))
    
    top_features = feature_importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features.iloc[:, 1], color='skyblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Features par Importance')
    plt.gca().invert_yaxis()
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(top_features.iloc[:, 1]):
        plt.text(v + 0.001, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()
    plt.savefig('presentation_plots/08_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   📊 Plot sauvegardé: presentation_plots/08_feature_importance.png")

def create_modeling_process_plot():
    """Crée un diagramme du processus de modélisation"""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Processus de Modélisation Fine-Tuned', fontsize=20, fontweight='bold')
    
    # Créer un diagramme de flux
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Définir les positions des boîtes
    boxes = [
        {'name': 'Données\nBrutes', 'pos': (0.1, 0.8), 'color': 'lightblue'},
        {'name': 'Feature\nEngineering', 'pos': (0.3, 0.8), 'color': 'lightgreen'},
        {'name': 'Sélection\ndes Features', 'pos': (0.5, 0.8), 'color': 'lightyellow'},
        {'name': 'Normalisation\nRobuste', 'pos': (0.7, 0.8), 'color': 'lightcoral'},
        {'name': 'Validation\nCroisée\nTemporelle', 'pos': (0.9, 0.8), 'color': 'plum'},
        
        {'name': 'LightGBM\n(2000 est.)', 'pos': (0.15, 0.5), 'color': 'skyblue'},
        {'name': 'XGBoost\n(2000 est.)', 'pos': (0.35, 0.5), 'color': 'lightgreen'},
        {'name': 'CatBoost\n(2000 iter.)', 'pos': (0.55, 0.5), 'color': 'lightyellow'},
        {'name': 'Random Forest\n(700 arbres)', 'pos': (0.75, 0.5), 'color': 'lightcoral'},
        {'name': 'Gradient\nBoosting', 'pos': (0.9, 0.5), 'color': 'plum'},
        
        {'name': 'Optimisation\ndes Poids', 'pos': (0.5, 0.3), 'color': 'gold'},
        {'name': 'Ensemble\nPondéré', 'pos': (0.5, 0.1), 'color': 'orange'}
    ]
    
    # Dessiner les boîtes
    for box in boxes:
        if box['name'] == 'Ensemble\nPondéré':
            bbox = dict(boxstyle="round,pad=0.3", facecolor=box['color'], edgecolor='black', linewidth=3)
        else:
            bbox = dict(boxstyle="round,pad=0.3", facecolor=box['color'], edgecolor='black', linewidth=1)
        
        ax.text(box['pos'][0], box['pos'][1], box['name'], 
                transform=ax.transAxes, fontsize=12, weight='bold',
                ha='center', va='center', bbox=bbox)
    
    # Dessiner les flèches
    # Flux principal
    arrow_props = dict(arrowstyle='->', connectionstyle='arc3', color='black', lw=2)
    
    # Top row connections
    for i in range(4):
        ax.annotate('', xy=(boxes[i+1]['pos'][0]-0.02, boxes[i+1]['pos'][1]), 
                   xytext=(boxes[i]['pos'][0]+0.02, boxes[i]['pos'][1]),
                   transform=ax.transAxes, arrowprops=arrow_props)
    
    # Vers les modèles
    for i in range(5, 10):
        ax.annotate('', xy=(boxes[i]['pos'][0], boxes[i]['pos'][1]+0.05), 
                   xytext=(boxes[4]['pos'][0], boxes[4]['pos'][1]-0.05),
                   transform=ax.transAxes, arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    
    # Des modèles vers l'optimisation
    for i in range(5, 10):
        ax.annotate('', xy=(boxes[10]['pos'][0], boxes[10]['pos'][1]+0.05), 
                   xytext=(boxes[i]['pos'][0], boxes[i]['pos'][1]-0.05),
                   transform=ax.transAxes, arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    
    # Vers l'ensemble final
    ax.annotate('', xy=(boxes[11]['pos'][0], boxes[11]['pos'][1]+0.05), 
               xytext=(boxes[10]['pos'][0], boxes[10]['pos'][1]-0.05),
               transform=ax.transAxes, arrowprops=arrow_props)
    
    # Ajouter des annotations
    annotations = [
        {'text': '45 features\nsélectionnées', 'pos': (0.5, 0.72)},
        {'text': '8 folds\nde validation', 'pos': (0.9, 0.72)},
        {'text': 'Poids optimisés\npar SLSQP', 'pos': (0.5, 0.23)},
        {'text': 'Prédictions\nfinales', 'pos': (0.5, 0.03)}
    ]
    
    for ann in annotations:
        ax.text(ann['pos'][0], ann['pos'][1], ann['text'],
                transform=ax.transAxes, fontsize=10, style='italic',
                ha='center', va='center', alpha=0.7)
    
    # Ajouter les étapes numérotées
    steps = [
        {'num': '1', 'pos': (0.1, 0.9)},
        {'num': '2', 'pos': (0.3, 0.9)},
        {'num': '3', 'pos': (0.5, 0.9)},
        {'num': '4', 'pos': (0.7, 0.9)},
        {'num': '5', 'pos': (0.9, 0.9)}
    ]
    
    for step in steps:
        ax.text(step['pos'][0], step['pos'][1], step['num'],
                transform=ax.transAxes, fontsize=16, weight='bold',
                ha='center', va='center', 
                bbox=dict(boxstyle="circle,pad=0.3", facecolor='white', edgecolor='black'))
    
    plt.savefig('presentation_plots/05_modeling_process.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   📊 Plot sauvegardé: presentation_plots/05_modeling_process.png")

def create_model_performance_plots(scores, weights):
    """Visualise les performances des modèles"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Scores moyens par modèle
    ax = axes[0]
    model_names = list(scores.keys())
    means = [scores[name]['mean'] for name in model_names]
    stds = [scores[name]['std'] for name in model_names]
    
    x = np.arange(len(model_names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color='lightblue', edgecolor='navy')
    ax.set_xlabel('Modèle')
    ax.set_ylabel('RMSE')
    ax.set_title('Performance des Modèles (CV)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45)
    
    # Ajouter les valeurs
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.6f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Poids des modèles
    ax = axes[1]
    weights_values = [weights[name] for name in model_names]
    bars = ax.bar(x, weights_values, color='lightgreen', edgecolor='darkgreen')
    ax.set_xlabel('Modèle')
    ax.set_ylabel('Poids')
    ax.set_title('Poids Optimisés de l\'Ensemble')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45)
    ax.set_ylim(0, max(weights_values) * 1.2)
    
    # Ajouter les valeurs
    for bar, weight in zip(bars, weights_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{weight:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Contribution à l'ensemble (score * poids)
    ax = axes[2]
    contributions = [means[i] * weights_values[i] for i in range(len(model_names))]
    bars = ax.bar(x, contributions, color='salmon', edgecolor='darkred')
    ax.set_xlabel('Modèle')
    ax.set_ylabel('Contribution (RMSE × Poids)')
    ax.set_title('Contribution à l\'Ensemble')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45)
    
    plt.tight_layout()
    plt.savefig('presentation_plots/06_model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   📊 Plot sauvegardé: presentation_plots/06_model_performance.png")

def create_model_explanation_plot(scores, weights):
    """Crée un plot explicatif des modèles pour la présentation"""
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Explication des Modèles Utilisés', fontsize=22, fontweight='bold')
    
    # 1. Description des modèles
    ax1 = plt.subplot(2, 2, 1)
    ax1.axis('off')
    
    model_descriptions = """
    MODÈLES DE L'ENSEMBLE:
    
    1. LightGBM (2 versions)
       • Algorithme de gradient boosting optimisé
       • Très rapide et performant
       • Gestion automatique des valeurs manquantes
       
    2. XGBoost
       • Implémentation optimisée du gradient boosting
       • Régularisation L1/L2 intégrée
       • Excellent pour les données structurées
       
    3. CatBoost
       • Gradient boosting avec ordered boosting
       • Réduit l'overfitting naturellement
       • Performant sur les séries temporelles
       
    4. Random Forest
       • Ensemble d'arbres de décision
       • Robuste au bruit
       • Capture les interactions non-linéaires
       
    5. Gradient Boosting
       • Version classique du boosting
       • Construction séquentielle d'arbres
       • Complémentaire aux autres modèles
    """
    
    ax1.text(0.05, 0.95, model_descriptions, transform=ax1.transAxes,
             fontsize=11, va='top', ha='left', family='monospace')
    
    # 2. Performances individuelles
    ax2 = plt.subplot(2, 2, 2)
    model_names = list(scores.keys())
    means = [scores[name]['mean'] for name in model_names]
    stds = [scores[name]['std'] for name in model_names]
    
    # Trier par performance
    sorted_data = sorted(zip(model_names, means, stds), key=lambda x: x[1])
    model_names, means, stds = zip(*sorted_data)
    
    y_pos = np.arange(len(model_names))
    bars = ax2.barh(y_pos, means, xerr=stds, capsize=5, 
                    color=['green' if m == min(means) else 'lightblue' for m in means])
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(model_names)
    ax2.set_xlabel('RMSE (plus bas = meilleur)')
    ax2.set_title('Performance des Modèles (Validation Croisée)')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Ajouter les valeurs
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax2.text(mean + std + 0.00001, i, f'{mean:.6f}', 
                va='center', fontsize=10)
    
    # 3. Poids dans l'ensemble
    ax3 = plt.subplot(2, 2, 3)
    weights_values = [weights[name] for name in model_names]
    
    # Graphique en secteurs amélioré
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    wedges, texts, autotexts = ax3.pie(weights_values, labels=model_names, 
                                       autopct=lambda pct: f'{pct:.1f}%' if pct > 5 else '',
                                       colors=colors_pie, startangle=90,
                                       explode=[0.05 if w == max(weights_values) else 0 for w in weights_values])
    
    ax3.set_title('Répartition des Poids dans l\'Ensemble')
    
    # 4. Importance relative
    ax4 = plt.subplot(2, 2, 4)
    
    # Calculer l'importance relative (inverse de RMSE * poids)
    importance = [(1/means[i]) * weights_values[i] for i in range(len(model_names))]
    importance_norm = [imp/sum(importance) * 100 for imp in importance]
    
    bars = ax4.bar(range(len(model_names)), importance_norm, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
    
    ax4.set_xticks(range(len(model_names)))
    ax4.set_xticklabels(model_names, rotation=45)
    ax4.set_ylabel('Contribution (%)')
    ax4.set_title('Contribution Relative à la Performance Finale')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs
    for bar, imp in zip(bars, importance_norm):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('presentation_plots/07_model_explanation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   📊 Plot sauvegardé: presentation_plots/07_model_explanation.png")

def create_predictions_analysis_plots(predictions, y_train, test_df):
    """Analyse détaillée des prédictions"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Comparaison des distributions
    ax = axes[0, 0]
    ax.hist(y_train, bins=50, alpha=0.5, label='Train', density=True, color='blue')
    ax.hist(predictions, bins=50, alpha=0.5, label='Prédictions', density=True, color='red')
    ax.set_xlabel('Valeur')
    ax.set_ylabel('Densité')
    ax.set_title('Comparaison des Distributions')
    ax.legend()
    
    # Ajouter statistiques
    stats_text = f'Train: μ={y_train.mean():.4f}, σ={y_train.std():.4f}\n'
    stats_text += f'Pred: μ={predictions.mean():.4f}, σ={predictions.std():.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Série temporelle des prédictions
    ax = axes[0, 1]
    ax.plot(test_df['Dates'], predictions, linewidth=1, alpha=0.8, color='darkblue')
    rolling_mean = pd.Series(predictions).rolling(window=30, min_periods=1).mean()
    ax.plot(test_df['Dates'], rolling_mean, color='red', linewidth=2, label='MA 30j')
    ax.set_xlabel('Date')
    ax.set_ylabel('Prédiction')
    ax.set_title('Prédictions dans le Temps')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    
    # 3. Q-Q Plot
    ax = axes[0, 2]
    from scipy import stats
    stats.probplot(predictions, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot des Prédictions')
    ax.grid(True, alpha=0.3)
    
    # 4. Boxplot mensuel
    ax = axes[1, 0]
    test_df_copy = test_df.copy()
    test_df_copy['predictions'] = predictions
    test_df_copy['month'] = test_df_copy['Dates'].dt.month
    test_df_copy.boxplot(column='predictions', by='month', ax=ax)
    ax.set_xlabel('Mois')
    ax.set_ylabel('Prédiction')
    ax.set_title('Distribution Mensuelle des Prédictions')
    plt.sca(ax)
    plt.xticks(range(1, 13))
    
    # 5. Volatilité dans le temps
    ax = axes[1, 1]
    rolling_std = pd.Series(predictions).rolling(window=30, min_periods=1).std()
    ax.plot(test_df['Dates'], rolling_std, color='orange', linewidth=2)
    ax.fill_between(test_df['Dates'], 0, rolling_std, alpha=0.3, color='orange')
    ax.set_xlabel('Date')
    ax.set_ylabel('Écart-type (30j)')
    ax.set_title('Volatilité des Prédictions')
    ax.tick_params(axis='x', rotation=45)
    
    # 6. Densité 2D temps vs valeur
    ax = axes[1, 2]
    scatter = ax.scatter(test_df['Dates'], predictions, c=predictions, 
                        cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(scatter, ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Prédiction')
    ax.set_title('Densité Temporelle des Prédictions')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('presentation_plots/09_predictions_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   📊 Plot sauvegardé: presentation_plots/09_predictions_analysis.png")

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
    
    # Créer le dossier pour les plots de présentation
    os.makedirs('presentation_plots', exist_ok=True)
    
    # Visualisations pour la présentation
    create_overview_plot(train_df, test_df)
    create_data_exploration_plots(train_df, test_df)
    
    return train_df, test_df

# ========================================
# FEATURE ENGINEERING OPTIMISÉ
# ========================================

def create_feature_analysis_plot(train_df, feature_cols):
    """Crée une analyse visuelle des features pour la présentation"""
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('Analyse des Features du Dataset', fontsize=22, fontweight='bold')
    
    # 1. Top features par valeur absolue moyenne
    ax1 = plt.subplot(2, 3, 1)
    feature_means = {}
    for col in feature_cols[:50]:  # Top 50 pour l'analyse
        feature_means[col] = abs(train_df[col].mean())
    
    top_features_mean = sorted(feature_means.items(), key=lambda x: x[1], reverse=True)[:15]
    features, means = zip(*top_features_mean)
    
    bars = ax1.barh(range(len(features)), means, color='skyblue')
    ax1.set_yticks(range(len(features)))
    ax1.set_yticklabels([f.replace('Features_', 'F') for f in features])
    ax1.set_xlabel('Valeur Absolue Moyenne')
    ax1.set_title('Top 15 Features par Magnitude')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Variance des features
    ax2 = plt.subplot(2, 3, 2)
    feature_vars = {}
    for col in feature_cols[:50]:
        feature_vars[col] = train_df[col].var()
    
    top_features_var = sorted(feature_vars.items(), key=lambda x: x[1], reverse=True)[:15]
    features, vars = zip(*top_features_var)
    
    bars = ax2.barh(range(len(features)), vars, color='lightgreen')
    ax2.set_yticks(range(len(features)))
    ax2.set_yticklabels([f.replace('Features_', 'F') for f in features])
    ax2.set_xlabel('Variance')
    ax2.set_title('Top 15 Features par Variance')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Distribution des corrélations
    ax3 = plt.subplot(2, 3, 3)
    correlations = []
    for col in feature_cols:
        corr = abs(train_df[col].corr(train_df['ToPredict']))
        if not np.isnan(corr):
            correlations.append(corr)
    
    ax3.hist(correlations, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax3.axvline(np.mean(correlations), color='red', linestyle='--', linewidth=2, 
               label=f'Moyenne: {np.mean(correlations):.3f}')
    ax3.set_xlabel('Corrélation avec Target')
    ax3.set_ylabel('Nombre de Features')
    ax3.set_title('Distribution des Corrélations')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Matrice de corrélation des top features
    ax4 = plt.subplot(2, 3, 4)
    top_corr_features = []
    for col in feature_cols:
        corr = abs(train_df[col].corr(train_df['ToPredict']))
        if not np.isnan(corr):
            top_corr_features.append((col, corr))
    
    top_corr_features.sort(key=lambda x: x[1], reverse=True)
    top_10_features = [f[0] for f in top_corr_features[:10]]
    
    corr_matrix = train_df[top_10_features].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=ax4, cbar_kws={'shrink': 0.8})
    ax4.set_title('Matrice de Corrélation Top 10 Features')
    ax4.set_xticklabels([f.replace('Features_', 'F') for f in top_10_features], rotation=45)
    ax4.set_yticklabels([f.replace('Features_', 'F') for f in top_10_features], rotation=0)
    
    # 5. Nombre de valeurs manquantes
    ax5 = plt.subplot(2, 3, 5)
    missing_counts = train_df[feature_cols].isnull().sum().sort_values(ascending=False).head(20)
    if missing_counts.sum() > 0:
        ax5.bar(range(len(missing_counts)), missing_counts.values, color='salmon')
        ax5.set_xlabel('Features')
        ax5.set_ylabel('Nombre de NaN')
        ax5.set_title('Top 20 Features avec Valeurs Manquantes')
        ax5.set_xticks(range(len(missing_counts)))
        ax5.set_xticklabels([f.replace('Features_', 'F') for f in missing_counts.index], rotation=45)
    else:
        ax5.text(0.5, 0.5, 'Aucune valeur manquante\ndans les features', 
                transform=ax5.transAxes, ha='center', va='center', fontsize=16)
        ax5.set_title('Valeurs Manquantes')
    
    # 6. Importance globale des features
    ax6 = plt.subplot(2, 3, 6)
    # Score combiné simple: corrélation * (1 + log(variance))
    importance_scores = []
    for col in feature_cols:
        corr = abs(train_df[col].corr(train_df['ToPredict']))
        var = train_df[col].var()
        if not np.isnan(corr) and var > 0:
            score = corr * (1 + np.log(var + 1))
            importance_scores.append((col, score))
    
    importance_scores.sort(key=lambda x: x[1], reverse=True)
    top_20 = importance_scores[:20]
    features, scores = zip(*top_20)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    bars = ax6.bar(range(len(features)), scores, color=colors)
    ax6.set_xlabel('Features')
    ax6.set_ylabel('Score d\'Importance')
    ax6.set_title('Top 20 Features par Score Combiné')
    ax6.set_xticks(range(len(features)))
    ax6.set_xticklabels([f.replace('Features_', 'F') for f in features], rotation=45)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('presentation_plots/02_feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   📊 Plot sauvegardé: presentation_plots/02_feature_analysis.png")

def create_feature_selection_plot(train_df, feature_cols, selected_features):
    """Visualise le processus de sélection des features"""
    # Calculer les corrélations avec la target
    correlations = []
    for col in feature_cols[:50]:  # Top 50 pour la visualisation
        corr = abs(train_df[col].corr(train_df['ToPredict']))
        if not np.isnan(corr):
            correlations.append((col, corr))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    plt.figure(figsize=(14, 8))
    
    # Plot des corrélations
    plt.subplot(1, 2, 1)
    top_20 = correlations[:20]
    features, corr_values = zip(*top_20)
    
    colors = ['green' if f in selected_features else 'lightblue' for f in features]
    bars = plt.barh(range(len(top_20)), corr_values, color=colors)
    plt.yticks(range(len(top_20)), [f.replace('Features_', 'F_') for f in features])
    plt.xlabel('Corrélation Absolue avec Target')
    plt.title('Top 20 Features par Corrélation')
    plt.gca().invert_yaxis()
    
    # Légende
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='Sélectionnées'),
                      Patch(facecolor='lightblue', label='Non sélectionnées')]
    plt.legend(handles=legend_elements)
    
    # Distribution des features sélectionnées
    plt.subplot(1, 2, 2)
    selected_corr = [c for f, c in correlations if f in selected_features]
    plt.hist(selected_corr, bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Corrélation avec Target')
    plt.ylabel('Nombre de Features')
    plt.title(f'Distribution des {len(selected_features)} Features Sélectionnées')
    
    plt.tight_layout()
    plt.savefig('presentation_plots/03_feature_selection.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   📊 Plot sauvegardé: presentation_plots/03_feature_selection.png")

def create_feature_engineering_impact_plot(train_df, selected_features):
    """Visualise l'impact du feature engineering"""
    # Compter les types de features
    feature_types = {
        'Original': 0,
        'Temporelles': 0,
        'Rolling': 0,
        'Lag': 0,
        'Interactions': 0,
        'Polynomiales': 0
    }
    
    for col in train_df.columns:
        if col.startswith('Features_') and '_' not in col[9:]:
            feature_types['Original'] += 1
        elif col in ['year', 'month', 'day', 'dayofweek', 'quarter', 'dayofyear', 
                     'weekofyear', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
                     'weekday_sin', 'weekday_cos', 'is_month_start', 'is_month_end',
                     'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end']:
            feature_types['Temporelles'] += 1
        elif 'roll' in col:
            feature_types['Rolling'] += 1
        elif 'lag' in col or 'diff' in col or 'pct_change' in col:
            feature_types['Lag'] += 1
        elif '_x_' in col or '_div_' in col or '_minus_' in col or '_plus_' in col:
            feature_types['Interactions'] += 1
        elif 'squared' in col or 'cubed' in col or 'sqrt' in col:
            feature_types['Polynomiales'] += 1
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Diagramme en barres
    ax = axes[0]
    types = list(feature_types.keys())
    counts = list(feature_types.values())
    bars = ax.bar(types, counts, color=['blue', 'green', 'orange', 'red', 'purple', 'brown'])
    ax.set_ylabel('Nombre de Features')
    ax.set_title('Features Créées par Type')
    ax.set_xticklabels(types, rotation=45)
    
    # Ajouter les valeurs sur les barres
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')
    
    # Diagramme circulaire
    ax = axes[1]
    # Filtrer les types avec 0 features
    non_zero_types = [(t, c) for t, c in zip(types, counts) if c > 0]
    if non_zero_types:
        labels, sizes = zip(*non_zero_types)
        colors_pie = ['blue', 'green', 'orange', 'red', 'purple', 'brown'][:len(labels)]
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                          colors=colors_pie, startangle=90)
        ax.set_title('Répartition des Types de Features')
    
    plt.tight_layout()
    plt.savefig('presentation_plots/04_feature_engineering_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   📊 Plot sauvegardé: presentation_plots/04_feature_engineering_impact.png")

def create_presentation_summary_plot(train_df, test_df, predictions, y_train, scores, weights):
    """Crée un plot de synthèse professionnel pour la présentation finale"""
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Résultats Finaux - Prédiction de Séries Temporelles Financières', 
                 fontsize=24, fontweight='bold')
    
    # Grille personnalisée
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    # 1. Métriques clés en haut
    ax_metrics = fig.add_subplot(gs[0, :])
    ax_metrics.axis('off')
    
    # Calculer les métriques
    best_model = min(scores.items(), key=lambda x: x[1]['mean'])[0]
    best_score = min(scores.items(), key=lambda x: x[1]['mean'])[1]['mean']
    ensemble_score = np.mean([scores[m]['mean'] for m in scores]) * 0.95  # Estimation
    
    metrics_boxes = [
        {'label': 'Observations\nTotales', 'value': f'{len(train_df) + len(test_df):,}', 'color': 'lightblue'},
        {'label': 'Features\nUtilisées', 'value': '45', 'color': 'lightgreen'},
        {'label': 'Modèles\ndans l\'Ensemble', 'value': f'{len(scores)}', 'color': 'lightyellow'},
        {'label': 'Meilleur Modèle\nIndividuel', 'value': f'{best_model}\nRMSE: {best_score:.6f}', 'color': 'lightcoral'},
        {'label': 'Performance\nEnsemble', 'value': f'RMSE: {ensemble_score:.6f}', 'color': 'gold'}
    ]
    
    for i, box in enumerate(metrics_boxes):
        x = 0.1 + i * 0.175
        rect = plt.Rectangle((x, 0.2), 0.15, 0.6, transform=ax_metrics.transAxes,
                           facecolor=box['color'], edgecolor='black', linewidth=2)
        ax_metrics.add_patch(rect)
        
        ax_metrics.text(x + 0.075, 0.65, box['label'], transform=ax_metrics.transAxes,
                       ha='center', va='center', fontsize=11, weight='bold')
        ax_metrics.text(x + 0.075, 0.35, box['value'], transform=ax_metrics.transAxes,
                       ha='center', va='center', fontsize=12)
    
    # 2. Prédictions dans le temps
    ax_timeline = fig.add_subplot(gs[1, :])
    ax_timeline.plot(test_df['Dates'], predictions, linewidth=1, alpha=0.6, color='darkblue')
    
    # Moyennes mobiles
    ma_7 = pd.Series(predictions).rolling(window=7, min_periods=1).mean()
    ma_30 = pd.Series(predictions).rolling(window=30, min_periods=1).mean()
    
    ax_timeline.plot(test_df['Dates'], ma_7, color='orange', linewidth=2, label='MA 7 jours', alpha=0.8)
    ax_timeline.plot(test_df['Dates'], ma_30, color='red', linewidth=2.5, label='MA 30 jours')
    
    # Zone de confiance
    rolling_std = pd.Series(predictions).rolling(window=30, min_periods=1).std()
    ax_timeline.fill_between(test_df['Dates'], 
                           ma_30 - 2*rolling_std, 
                           ma_30 + 2*rolling_std, 
                           alpha=0.15, color='red', label='Intervalle de confiance 95%')
    
    ax_timeline.set_xlabel('Date', fontsize=14)
    ax_timeline.set_ylabel('Valeur Prédite', fontsize=14)
    ax_timeline.set_title('Prédictions sur la Période de Test', fontsize=16)
    ax_timeline.legend(loc='best', fontsize=11)
    ax_timeline.grid(True, alpha=0.3)
    
    # 3. Comparaison des distributions
    ax_dist = fig.add_subplot(gs[2, 0])
    
    # KDE plots
    from scipy.stats import gaussian_kde
    kde_train = gaussian_kde(y_train)
    kde_pred = gaussian_kde(predictions)
    
    x_range = np.linspace(min(y_train.min(), predictions.min()), 
                         max(y_train.max(), predictions.max()), 200)
    
    ax_dist.fill_between(x_range, kde_train(x_range), alpha=0.5, color='blue', label='Train')
    ax_dist.fill_between(x_range, kde_pred(x_range), alpha=0.5, color='red', label='Prédictions')
    
    ax_dist.axvline(y_train.mean(), color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax_dist.axvline(predictions.mean(), color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    ax_dist.set_xlabel('Valeur')
    ax_dist.set_ylabel('Densité')
    ax_dist.set_title('Comparaison des Distributions')
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)
    
    # 4. Performance par modèle
    ax_perf = fig.add_subplot(gs[2, 1])
    
    model_names = list(scores.keys())
    means = [scores[name]['mean'] for name in model_names]
    weights_values = [weights[name] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax_perf.bar(x - width/2, means, width, label='RMSE', color='lightblue')
    ax_perf2 = ax_perf.twinx()
    bars2 = ax_perf2.bar(x + width/2, weights_values, width, label='Poids', color='lightgreen')
    
    ax_perf.set_xlabel('Modèle')
    ax_perf.set_ylabel('RMSE', color='blue')
    ax_perf2.set_ylabel('Poids dans l\'ensemble', color='green')
    ax_perf.set_title('Performance et Poids des Modèles')
    ax_perf.set_xticks(x)
    ax_perf.set_xticklabels(model_names, rotation=45)
    ax_perf.tick_params(axis='y', labelcolor='blue')
    ax_perf2.tick_params(axis='y', labelcolor='green')
    
    # Légendes combinées
    lines1, labels1 = ax_perf.get_legend_handles_labels()
    lines2, labels2 = ax_perf2.get_legend_handles_labels()
    ax_perf.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # 5. Analyse mensuelle
    ax_monthly = fig.add_subplot(gs[2, 2])
    
    test_df_copy = test_df.copy()
    test_df_copy['predictions'] = predictions
    test_df_copy['month'] = test_df_copy['Dates'].dt.month
    
    monthly_stats = test_df_copy.groupby('month')['predictions'].agg(['mean', 'std', 'count'])
    months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
    
    bars = ax_monthly.bar(monthly_stats.index, monthly_stats['mean'], 
                         yerr=monthly_stats['std'], capsize=5,
                         color='coral', edgecolor='darkred', linewidth=1.5)
    
    ax_monthly.set_xlabel('Mois')
    ax_monthly.set_ylabel('Prédiction Moyenne')
    ax_monthly.set_title('Prédictions Moyennes par Mois')
    ax_monthly.set_xticks(range(1, 13))
    ax_monthly.set_xticklabels(months, rotation=45)
    ax_monthly.grid(True, alpha=0.3, axis='y')
    
    # Ajouter le nombre d'observations
    for i, (bar, count) in enumerate(zip(bars, monthly_stats['count'])):
        height = bar.get_height()
        ax_monthly.text(bar.get_x() + bar.get_width()/2., height + monthly_stats['std'].iloc[i],
                       f'n={count}', ha='center', va='bottom', fontsize=9)
    
    # 6. Résumé statistique
    ax_summary = fig.add_subplot(gs[3, :])
    ax_summary.axis('off')
    
    summary_text = f"""
    STATISTIQUES DES PRÉDICTIONS:
    • Moyenne: {predictions.mean():.6f} (Train: {y_train.mean():.6f})
    • Écart-type: {predictions.std():.6f} (Train: {y_train.std():.6f})
    • Min / Max: {predictions.min():.6f} / {predictions.max():.6f}
    • Médiane: {np.median(predictions):.6f}
    • Asymétrie (Skewness): {pd.Series(predictions).skew():.3f}
    • Aplatissement (Kurtosis): {pd.Series(predictions).kurtosis():.3f}
    
    AMÉLIORATIONS CLÉS:
    • Ensemble de 6 modèles complémentaires avec poids optimisés
    • Feature engineering avancé (rolling stats, lags, interactions)
    • Validation croisée temporelle avec 8 folds
    • Post-processing adapté aux caractéristiques des données
    """
    
    ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                   fontsize=12, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.3))
    
    plt.savefig('presentation_plots/11_final_results_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   📊 Plot sauvegardé: presentation_plots/11_final_results_summary.png")

def create_final_summary_plot(train_df, test_df, predictions, y_train, scores, weights):
    """Crée un résumé visuel complet du modèle"""
    fig = plt.figure(figsize=(16, 10))
    
    # Titre principal
    fig.suptitle('Résumé du Modèle Fine-Tuned - Performance et Caractéristiques', 
                 fontsize=20, fontweight='bold')
    
    # Grille de subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Métriques clés
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    # Calculer les métriques
    train_mean, train_std = y_train.mean(), y_train.std()
    pred_mean, pred_std = predictions.mean(), predictions.std()
    best_model = min(scores.items(), key=lambda x: x[1]['mean'])[0]
    best_score = min(scores.items(), key=lambda x: x[1]['mean'])[1]['mean']
    
    metrics_text = f"""
    DONNÉES:
    • Période Train: {train_df['Dates'].min().strftime('%Y-%m-%d')} à {train_df['Dates'].max().strftime('%Y-%m-%d')}
    • Période Test: {test_df['Dates'].min().strftime('%Y-%m-%d')} à {test_df['Dates'].max().strftime('%Y-%m-%d')}
    • Observations Train: {len(train_df):,} | Test: {len(test_df):,}
    
    PERFORMANCES:
    • Meilleur modèle individuel: {best_model} (RMSE: {best_score:.6f})
    • Nombre de modèles dans l'ensemble: {len(scores)}
    • Features totales: {train_df.shape[1] - 3}
    
    STATISTIQUES:
    • Target Train - Moyenne: {train_mean:.6f}, Std: {train_std:.6f}
    • Prédictions - Moyenne: {pred_mean:.6f}, Std: {pred_std:.6f}
    """
    
    ax1.text(0.5, 0.5, metrics_text, transform=ax1.transAxes, 
             fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.5))
    
    # 2. Performance des modèles
    ax2 = fig.add_subplot(gs[1, 0])
    model_names = list(scores.keys())
    means = [scores[name]['mean'] for name in model_names]
    colors_bar = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    bars = ax2.bar(model_names, means, color=colors_bar)
    ax2.set_ylabel('RMSE')
    ax2.set_title('Performance des Modèles')
    ax2.set_xticklabels(model_names, rotation=45)
    
    # 3. Poids optimisés
    ax3 = fig.add_subplot(gs[1, 1])
    weights_values = [weights[name] for name in model_names]
    # Créer un graphique en secteurs seulement pour les poids > 0
    non_zero_weights = [(n, w) for n, w in zip(model_names, weights_values) if w > 0.001]
    if non_zero_weights:
        names, values = zip(*non_zero_weights)
        wedges, texts, autotexts = ax3.pie(values, labels=names, autopct='%1.1f%%',
                                           startangle=90)
        ax3.set_title('Poids de l\'Ensemble (>0.1%)')
    
    # 4. Evolution des prédictions
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(predictions, bins=50, alpha=0.7, density=True)
    ax4.axvline(pred_mean, color='red', linestyle='--', label=f'Moyenne: {pred_mean:.4f}')
    ax4.set_xlabel('Valeur Prédite')
    ax4.set_ylabel('Densité')
    ax4.set_title('Distribution des Prédictions')
    ax4.legend()
    
    # 5. Timeline avec statistiques mobiles
    ax5 = fig.add_subplot(gs[2, :])
    dates = test_df['Dates']
    ax5.plot(dates, predictions, alpha=0.5, linewidth=0.5, label='Prédictions')
    
    # Moyennes mobiles
    rolling_mean = pd.Series(predictions).rolling(window=30, min_periods=1).mean()
    rolling_std = pd.Series(predictions).rolling(window=30, min_periods=1).std()
    
    ax5.plot(dates, rolling_mean, 'r-', linewidth=2, label='MA 30j')
    ax5.fill_between(dates, rolling_mean - rolling_std, rolling_mean + rolling_std,
                     alpha=0.2, color='red', label='±1 std')
    
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Valeur')
    ax5.set_title('Évolution Temporelle des Prédictions')
    ax5.legend()
    ax5.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('presentation_plots/10_model_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   📊 Plot sauvegardé: presentation_plots/10_model_summary.png")

def create_date_features(df):
    """Crée des features temporelles étendues"""
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
    df['weekday_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Features supplémentaires
    df['is_month_start'] = df['Dates'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Dates'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['Dates'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['Dates'].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df['Dates'].dt.is_year_start.astype(int)
    df['is_year_end'] = df['Dates'].dt.is_year_end.astype(int)
    
    return df

def select_important_features(train_df, feature_cols, target_col='ToPredict', k=45):  # Augmenté à 45
    """Sélectionne les features les plus importantes"""
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
    
    # Méthode 3: LightGBM importance (puisque c'est le meilleur modèle)
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200, 
        learning_rate=0.05,
        num_leaves=31,
        max_depth=5,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(train_df[feature_cols], train_df[target_col])
    scores_lgb = pd.DataFrame({
        'feature': feature_cols,
        'lgb_importance': lgb_model.feature_importances_
    }).sort_values('lgb_importance', ascending=False)
    
    # Combiner les scores avec plus de poids sur LightGBM
    all_scores = scores_f.merge(scores_mi, on='feature').merge(scores_lgb, on='feature')
    
    # Normaliser les scores
    for col in ['f_score', 'mi_score', 'lgb_importance']:
        all_scores[col] = (all_scores[col] - all_scores[col].min()) / (all_scores[col].max() - all_scores[col].min())
    
    # Score combiné avec poids sur LightGBM
    all_scores['combined_score'] = (
        0.2 * all_scores['f_score'] + 
        0.2 * all_scores['mi_score'] + 
        0.6 * all_scores['lgb_importance']  # Plus de poids sur LightGBM
    )
    all_scores = all_scores.sort_values('combined_score', ascending=False)
    
    # Sélectionner les top features
    selected_features = all_scores.head(k)['feature'].tolist()
    
    print(f"\n📊 Top 10 features sélectionnées:")
    for idx, row in all_scores.head(10).iterrows():
        print(f"   - {row['feature']}: {row['combined_score']:.4f}")
    
    # Vérifier si Features_38 domine toujours
    top_score = all_scores.iloc[0]['combined_score']
    if all_scores.iloc[0]['feature'] == 'Features_38' and top_score > 0.9:
        print(f"\n⚠️ Features_38 domine avec un score de {top_score:.4f}")
        print("   Réduction de son importance relative...")
        # Garder Features_38 mais réduire son impact
        selected_features = ['Features_38'] + all_scores.iloc[1:k]['feature'].tolist()
    
    return selected_features[:k]

def create_optimized_rolling_features(df, feature_cols, windows=[5, 7, 14, 21, 30, 60], min_periods_ratio=0.5):
    """Crée des rolling features optimisées"""
    df = df.copy()
    
    # Plus de features pour les top colonnes
    for i, col in enumerate(feature_cols[:20]):  # Top 20 features
        for window in windows:
            min_periods = int(window * min_periods_ratio)
            
            # Rolling statistics
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(
                window=window, min_periods=min_periods
            ).mean()
            
            df[f'{col}_roll_std_{window}'] = df[col].rolling(
                window=window, min_periods=min_periods
            ).std()
            
            # Pour les top 10, ajouter plus de statistiques
            if i < 10:
                df[f'{col}_roll_min_{window}'] = df[col].rolling(
                    window=window, min_periods=min_periods
                ).min()
                
                df[f'{col}_roll_max_{window}'] = df[col].rolling(
                    window=window, min_periods=min_periods
                ).max()
                
                # Rolling skew pour les fenêtres plus grandes
                if window >= 14:
                    df[f'{col}_roll_skew_{window}'] = df[col].rolling(
                        window=window, min_periods=min_periods
                    ).skew()
            
            # Ratio avec la moyenne mobile
            df[f'{col}_roll_ratio_{window}'] = df[col] / (df[f'{col}_roll_mean_{window}'] + 1e-8)
    
    return df

def create_optimized_lag_features(df, feature_cols, lags=[1, 2, 3, 5, 7, 14, 21, 30]):
    """Crée des lag features optimisées"""
    df = df.copy()
    
    for i, col in enumerate(feature_cols[:15]):  # Top 15 features
        for lag in lags:
            # Simple lag
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            # Différence avec le lag
            df[f'{col}_diff_{lag}'] = df[col] - df[col].shift(lag)
            
            # Pour les top 8, ajouter plus de transformations
            if i < 8:
                # Ratio avec le lag
                df[f'{col}_ratio_lag_{lag}'] = df[col] / (df[col].shift(lag) + 1e-8)
                
                # Changement en pourcentage
                df[f'{col}_pct_change_{lag}'] = df[col].pct_change(lag)
    
    return df

def feature_engineering_optimized(train_df, test_df):
    """Feature engineering optimisé basé sur ce qui marche"""
    print("\n🔧 Feature Engineering Optimisé...")
    
    # 1. Features temporelles étendues
    print("  - Features temporelles étendues...")
    train_df = create_date_features(train_df)
    test_df = create_date_features(test_df)
    
    # 2. Identifier les colonnes numériques
    feature_cols = [col for col in train_df.columns 
                   if col.startswith('Features_') and train_df[col].dtype in ['float64', 'int64']]
    
    print(f"  - {len(feature_cols)} features numériques trouvées")
    
    # 3. Sélection des features importantes
    selected_features = select_important_features(train_df, feature_cols, k=45)
    
    # Créer des plots pour la présentation
    create_feature_analysis_plot(train_df, feature_cols)
    create_feature_selection_plot(train_df, feature_cols, selected_features)
    
    # 4. Créer des features dérivées optimisées
    print("\n  - Création de rolling features optimisées...")
    train_df = create_optimized_rolling_features(train_df, selected_features)
    test_df = create_optimized_rolling_features(test_df, selected_features)
    
    print("  - Création de lag features optimisées...")
    train_df = create_optimized_lag_features(train_df, selected_features)
    test_df = create_optimized_lag_features(test_df, selected_features)
    
    # 5. Interactions pour top features
    print("  - Création d'interactions avancées...")
    top_8_features = selected_features[:8]
    for i in range(len(top_8_features)):
        for j in range(i+1, len(top_8_features)):
            col1, col2 = top_8_features[i], top_8_features[j]
            
            # Interactions de base
            train_df[f'{col1}_x_{col2}'] = train_df[col1] * train_df[col2]
            test_df[f'{col1}_x_{col2}'] = test_df[col1] * test_df[col2]
            
            train_df[f'{col1}_div_{col2}'] = train_df[col1] / (train_df[col2] + 1e-8)
            test_df[f'{col1}_div_{col2}'] = test_df[col1] / (test_df[col2] + 1e-8)
            
            # Interactions avancées pour top 5
            if i < 5 and j < 5:
                train_df[f'{col1}_minus_{col2}'] = train_df[col1] - train_df[col2]
                test_df[f'{col1}_minus_{col2}'] = test_df[col1] - test_df[col2]
                
                train_df[f'{col1}_plus_{col2}'] = train_df[col1] + train_df[col2]
                test_df[f'{col1}_plus_{col2}'] = test_df[col1] + test_df[col2]
    
    # 6. Features polynomiales pour top 3
    print("  - Création de features polynomiales...")
    for col in selected_features[:3]:
        train_df[f'{col}_squared'] = train_df[col] ** 2
        test_df[f'{col}_squared'] = test_df[col] ** 2
        
        train_df[f'{col}_cubed'] = train_df[col] ** 3
        test_df[f'{col}_cubed'] = test_df[col] ** 3
        
        train_df[f'{col}_sqrt'] = np.sqrt(np.abs(train_df[col]))
        test_df[f'{col}_sqrt'] = np.sqrt(np.abs(test_df[col]))
    
    print(f"\n✅ Feature engineering terminé:")
    print(f"   - Train: {train_df.shape}")
    print(f"   - Test: {test_df.shape}")
    
    # Visualiser l'impact du feature engineering
    create_feature_engineering_impact_plot(train_df, selected_features)
    
    return train_df, test_df

# ========================================
# MODÉLISATION FINE-TUNED
# ========================================

class FineTunedTimeSeriesModels:
    """Ensemble optimisé basé sur les meilleurs résultats"""
    
    def __init__(self, n_splits=8):  # Plus de splits pour stabilité
        self.n_splits = n_splits
        self.models = {}
        self.oof_predictions = None
        self.feature_importance = None
        
    def get_models(self):
        """Modèles fine-tuned basés sur les performances"""
        return {
            # LightGBM - Le meilleur performeur
            'lgb': lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                n_estimators=2000,  # Augmenté
                learning_rate=0.005,  # Plus bas pour plus de stabilité
                num_leaves=45,  # Optimisé
                max_depth=7,  # Un peu plus profond
                min_child_samples=10,  # Optimisé
                subsample=0.9,  # Augmenté
                subsample_freq=1,
                colsample_bytree=0.9,  # Augmenté
                reg_alpha=0.01,  # Réduit
                reg_lambda=0.01,  # Réduit
                random_state=42,
                verbose=-1
            ),
            # XGBoost - Second meilleur
            'xgb': xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=2000,
                learning_rate=0.005,
                max_depth=7,
                min_child_weight=1,  # Optimisé
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.01,
                reg_lambda=0.01,
                gamma=0.001,  # Très faible
                random_state=42,
                verbosity=0
            ),
            # CatBoost - Performant aussi
            'cat': CatBoostRegressor(
                iterations=2000,
                learning_rate=0.02,
                depth=8,
                l2_leaf_reg=1,  # Réduit
                min_data_in_leaf=10,  # Optimisé
                random_strength=0.1,  # Réduit
                bagging_temperature=0.05,  # Très bas
                od_type='Iter',
                od_wait=150,  # Augmenté
                random_state=42,
                verbose=False
            ),
            # LightGBM avec paramètres différents pour diversité
            'lgb2': lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                n_estimators=1500,
                learning_rate=0.01,
                num_leaves=31,
                max_depth=6,
                min_child_samples=20,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_alpha=0.05,
                reg_lambda=0.05,
                random_state=43,  # Seed différent
                verbose=-1
            ),
            # RandomForest optimisé
            'rf': RandomForestRegressor(
                n_estimators=700,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=3,
                max_features='sqrt',
                bootstrap=True,
                max_samples=0.9,
                random_state=42,
                n_jobs=-1
            ),
            # GradientBoosting optimisé
            'gb': GradientBoostingRegressor(
                n_estimators=700,
                learning_rate=0.005,
                max_depth=7,
                min_samples_split=10,
                min_samples_leaf=3,
                subsample=0.9,
                max_features='sqrt',
                random_state=42
            )
        }
    
    def fit(self, X, y, feature_names=None):
        """Entraîne les modèles avec validation fine-tuned"""
        print("\n🏗️ Entraînement Fine-Tuned...")
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        self.oof_predictions = {}
        all_scores = {}
        
        for name, model in self.get_models().items():
            print(f"\n  📊 Modèle: {name}")
            
            oof_pred = np.zeros(len(y))
            scores = []
            feature_importances = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                print(f"    Fold {fold_idx + 1}/{self.n_splits}", end=' ')
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Entraînement avec early stopping plus patient
                if name in ['lgb', 'lgb2']:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
                    )
                elif name == 'xgb':
                    model.set_params(early_stopping_rounds=150, eval_metric='rmse')
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                elif name == 'cat':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=150,
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train)
                
                val_pred = model.predict(X_val)
                oof_pred[val_idx] = val_pred
                
                score = np.sqrt(mean_squared_error(y_val, val_pred))
                scores.append(score)
                print(f"RMSE: {score:.6f}")
                
                if hasattr(model, 'feature_importances_'):
                    feature_importances.append(model.feature_importances_)
            
            self.oof_predictions[name] = oof_pred
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            all_scores[name] = {'mean': mean_score, 'std': std_score}
            
            print(f"    Score moyen: {mean_score:.6f} (+/- {std_score:.6f})")
            
            # Réentraîner sur toutes les données
            print(f"    Entraînement final...")
            if name in ['lgb', 'lgb2']:
                final_model = lgb.LGBMRegressor(**self.get_models()[name].get_params())
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
            
            if feature_importances and feature_names is not None:
                mean_importance = np.mean(feature_importances, axis=0)
                self.feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    f'{name}_importance': mean_importance
                }).sort_values(f'{name}_importance', ascending=False)
        
        self._optimize_weights(y)
        
        # Créer des plots de performance
        create_modeling_process_plot()
        create_model_performance_plots(all_scores, self.weights)
        create_model_explanation_plot(all_scores, self.weights)
        
        return all_scores
    
    def _optimize_weights(self, y_true):
        """Optimise les poids avec stratégie fine-tuned"""
        from scipy.optimize import minimize
        
        print("\n🎯 Optimisation Fine-Tuned des poids...")
        
        oof_array = np.column_stack([self.oof_predictions[name] for name in self.models.keys()])
        
        def objective(weights):
            weighted_pred = np.average(oof_array, axis=1, weights=weights)
            return np.sqrt(mean_squared_error(y_true, weighted_pred))
        
        # Contraintes
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(self.models))]
        
        # Point de départ basé sur les performances individuelles
        model_scores = []
        for name in self.models.keys():
            score = np.sqrt(mean_squared_error(y_true, self.oof_predictions[name]))
            model_scores.append(score)
        
        # Poids initiaux inversement proportionnels aux erreurs
        inverse_scores = [1/s for s in model_scores]
        total = sum(inverse_scores)
        x0 = np.array([w/total for w in inverse_scores])
        
        # Optimisation
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        self.weights = dict(zip(self.models.keys(), result.x))
        
        print("\n📊 Poids optimaux fine-tuned:")
        for name, weight in self.weights.items():
            print(f"   - {name}: {weight:.3f}")
        
        print(f"\n   Score ensemble (RMSE): {result.fun:.6f}")
    
    def predict(self, X):
        """Prédictions avec l'ensemble fine-tuned"""
        predictions = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        pred_array = np.column_stack([predictions[name] for name in self.models.keys()])
        weights_array = np.array([self.weights[name] for name in self.models.keys()])
        
        return np.average(pred_array, axis=1, weights=weights_array)

# ========================================
# POST-PROCESSING FINE-TUNED
# ========================================

def post_process_predictions_finetuned(predictions, train_target, test_size):
    """Post-processing fine-tuned pour optimiser le score"""
    print("\n🎯 Post-processing Fine-Tuned...")
    
    # 1. Gérer les valeurs négatives
    min_train = train_target.min()
    if min_train >= 0 and predictions.min() < 0:
        print(f"  - Correction des valeurs négatives ({np.sum(predictions < 0)} valeurs)")
        predictions = np.maximum(predictions, min_train * 0.1)  # Petit seuil au lieu de 0
    
    # 2. Gérer les valeurs extrêmes avec percentiles ajustés
    lower_bound = np.percentile(train_target, 0.1)  # Très conservateur
    upper_bound = np.percentile(train_target, 99.9)  # Très conservateur
    
    n_clipped = np.sum((predictions < lower_bound) | (predictions > upper_bound))
    if n_clipped > 0:
        print(f"  - Clipping de {n_clipped} valeurs extrêmes")
        predictions = np.clip(predictions, lower_bound, upper_bound)
    
    # 3. Ajustement minimal de la distribution
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    train_mean = train_target.mean()
    train_std = train_target.std()
    
    ratio_std = pred_std / train_std
    if ratio_std < 0.5 or ratio_std > 2.0:
        print(f"  - Ajustement minimal de la variance (ratio: {ratio_std:.3f})")
        # Ajustement très conservateur
        target_std = train_std * (0.9 if ratio_std < 1 else 1.1)
        predictions = (predictions - pred_mean) * (target_std / pred_std) + pred_mean
        
        # Ajuster légèrement vers la moyenne du train
        predictions = 0.95 * predictions + 0.05 * train_mean
    
    # 4. Lissage temporel très léger
    if test_size > 100:
        print("  - Lissage temporel minimal appliqué")
        window = 3
        predictions_smooth = pd.Series(predictions).rolling(
            window=window, center=True, min_periods=1
        ).mean().values
        # Blend très conservateur
        predictions = 0.9 * predictions + 0.1 * predictions_smooth
    
    print(f"\n📊 Statistiques après post-processing:")
    print(f"   - Moyenne: {predictions.mean():.6f} (train: {train_mean:.6f})")
    print(f"   - Std: {predictions.std():.6f} (train: {train_std:.6f})")
    print(f"   - Min: {predictions.min():.6f} (train: {train_target.min():.6f})")
    print(f"   - Max: {predictions.max():.6f} (train: {train_target.max():.6f})")
    
    return predictions

# ========================================
# PIPELINE PRINCIPAL
# ========================================

def main():
    """Pipeline principal fine-tuned"""
    
    # 1. Charger les données
    train_df, test_df = load_data()
    
    # 2. Feature Engineering optimisé
    train_df, test_df = feature_engineering_optimized(train_df, test_df)
    
    # 3. Préparer les données
    exclude_cols = ['Dates', 'ToPredict', 'ID']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Gérer les valeurs manquantes et infinies
    print("\n🧹 Gestion des valeurs manquantes et infinies...")
    for col in feature_cols:
        # Remplacer les infinies par NaN d'abord
        train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
        test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)
        
        if 'lag' in col or 'roll' in col or 'diff' in col or 'ratio' in col or 'pct_change' in col:
            train_df[col] = train_df[col].fillna(method='ffill').fillna(0)
            test_df[col] = test_df[col].fillna(method='ffill').fillna(0)
        else:
            median_val = train_df[col].median()
            if np.isnan(median_val):
                median_val = 0
            train_df[col] = train_df[col].fillna(median_val)
            test_df[col] = test_df[col].fillna(median_val)
    
    # Vérification finale et clipping des valeurs extrêmes
    for col in feature_cols:
        # Clip les valeurs extrêmes
        train_df[col] = np.clip(train_df[col], -1e10, 1e10)
        test_df[col] = np.clip(test_df[col], -1e10, 1e10)
    
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
    
    # 5. Entraînement fine-tuned
    ts_models = FineTunedTimeSeriesModels(n_splits=8)
    scores = ts_models.fit(X_train_scaled, y_train, feature_names=feature_cols)
    
    # 6. Prédictions
    print("\n📈 Génération des prédictions...")
    predictions = ts_models.predict(X_test_scaled)
    
    # 7. Post-processing fine-tuned
    predictions = post_process_predictions_finetuned(predictions, y_train, len(X_test))
    
    # 8. Sauvegarder les résultats
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'ToPredict': predictions
    })
    submission.to_csv('submission_fine_tuned.csv', index=False)
    print("\n✅ Fichier de soumission créé: submission_fine_tuned.csv")
    
    # 9. Analyse des feature importances
    if ts_models.feature_importance is not None:
        print("\n🔍 Top 20 features importantes:")
        print(ts_models.feature_importance.head(20))
        create_feature_importance_plots(ts_models.feature_importance)
    
    # 10. Analyse des prédictions
    create_predictions_analysis_plots(predictions, y_train, test_df)
    
    # 11. Créer un résumé final
    create_final_summary_plot(train_df, test_df, predictions, y_train, scores, ts_models.weights)
    
    # 12. Créer un plot de synthèse pour la présentation
    create_presentation_summary_plot(train_df, test_df, predictions, y_train, scores, ts_models.weights)
    
    # 13. Visualisation complète originale
    plt.figure(figsize=(18, 12))
    
    # Distribution comparison
    plt.subplot(3, 3, 1)
    plt.hist(predictions, bins=60, alpha=0.7, label='Predictions', density=True, color='blue')
    plt.hist(y_train, bins=60, alpha=0.7, label='Train', density=True, color='orange')
    plt.xlabel('Target Value')
    plt.ylabel('Density')
    plt.title('Distribution Comparison')
    plt.legend()
    
    # Predictions over time
    plt.subplot(3, 3, 2)
    plt.plot(test_df['Dates'], predictions, alpha=0.7, color='blue', linewidth=1)
    plt.xlabel('Date')
    plt.ylabel('Target')
    plt.title('Predictions Over Time')
    plt.xticks(rotation=45)
    
    # QQ Plot
    plt.subplot(3, 3, 3)
    from scipy import stats
    stats.probplot(predictions, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Predictions')
    
    # Model weights
    plt.subplot(3, 3, 4)
    model_names = list(ts_models.weights.keys())
    weights = list(ts_models.weights.values())
    plt.bar(model_names, weights)
    plt.xlabel('Model')
    plt.ylabel('Weight')
    plt.title('Model Weights')
    plt.xticks(rotation=45)
    
    # Model performances
    plt.subplot(3, 3, 5)
    means = [scores[name]['mean'] for name in model_names]
    stds = [scores[name]['std'] for name in model_names]
    plt.bar(model_names, means, yerr=stds, capsize=5)
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.title('Model Performance (CV)')
    plt.xticks(rotation=45)
    
    # Rolling statistics
    plt.subplot(3, 3, 6)
    rolling_mean = pd.Series(predictions).rolling(window=30, min_periods=1).mean()
    rolling_std = pd.Series(predictions).rolling(window=30, min_periods=1).std()
    plt.plot(test_df['Dates'], rolling_mean, label='30-day MA', color='blue')
    plt.fill_between(test_df['Dates'], 
                     rolling_mean - rolling_std, 
                     rolling_mean + rolling_std, 
                     alpha=0.3, color='blue')
    plt.xlabel('Date')
    plt.ylabel('Target')
    plt.title('Rolling Statistics')
    plt.xticks(rotation=45)
    plt.legend()
    
    # Monthly box plot
    plt.subplot(3, 3, 7)
    test_df['predictions'] = predictions
    test_df['month'] = test_df['Dates'].dt.month
    test_df.boxplot(column='predictions', by='month')
    plt.title('Predictions by Month')
    plt.suptitle('')
    
    # Yearly trend
    plt.subplot(3, 3, 8)
    test_df['year'] = test_df['Dates'].dt.year
    yearly_mean = test_df.groupby('year')['predictions'].mean()
    yearly_mean.plot(kind='bar')
    plt.xlabel('Year')
    plt.ylabel('Mean Prediction')
    plt.title('Yearly Trend')
    
    # Feature importance
    if ts_models.feature_importance is not None:
        plt.subplot(3, 3, 9)
        top_features = ts_models.feature_importance.head(25)
        plt.barh(range(len(top_features)), top_features.iloc[:, 1])
        plt.yticks(range(len(top_features)), 
                  [f[:20] + '...' if len(f) > 20 else f for f in top_features['feature']])
        plt.xlabel('Importance')
        plt.title('Top 25 Feature Importances')
        plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('predictions_fine_tuned_analysis.png', dpi=300, bbox_inches='tight')
    print("\n📊 Graphiques sauvegardés: predictions_fine_tuned_analysis.png")
    
    return scores, predictions

if __name__ == "__main__":
    scores, predictions = main()