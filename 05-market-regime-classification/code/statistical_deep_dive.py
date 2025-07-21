"""
ANALYSE STATISTIQUE APPROFONDIE
================================
Tests statistiques et visualisations complémentaires
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

print("="*80)
print("ANALYSE STATISTIQUE APPROFONDIE")
print("="*80)

# 1. CHARGEMENT
print("\n[1] Chargement des données...")
train = pd.read_csv('train.csv', parse_dates=['Date'])
test = pd.read_csv('test.csv', parse_dates=['Date'])
submission = pd.read_csv('submission.csv')

price_cols = [col for col in train.columns if col not in ['Date', 'Market_Regime']]

# 2. TESTS STATISTIQUES
print("\n[2] Tests statistiques...")

fig, axes = plt.subplots(3, 2, figsize=(18, 16))

# 2.1 Test de normalité des returns par régime
ax = axes[0, 0]
returns_by_regime = {}
for regime in [-1, 0, 1]:
    regime_data = train[train['Market_Regime'] == regime]
    all_returns = []
    for col in price_cols[:10]:
        returns = regime_data[col].pct_change().dropna()
        all_returns.extend(returns)
    returns_by_regime[regime] = all_returns

# Q-Q plots
for i, (regime, returns) in enumerate(returns_by_regime.items()):
    ax_qq = plt.subplot(3, 6, i*6 + 1)
    stats.probplot(returns[:1000], dist="norm", plot=ax_qq)
    ax_qq.set_title(f'Q-Q Plot Régime {regime}', fontsize=10)
    
# Box plots des returns
ax = axes[0, 0]
data_to_plot = [returns_by_regime[-1][:1000], returns_by_regime[0][:1000], returns_by_regime[1][:1000]]
bp = ax.boxplot(data_to_plot, labels=['Baissier', 'Neutre', 'Haussier'], 
                patch_artist=True, showfliers=False)

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title('Distribution des Returns par Régime (sans outliers)', fontsize=14, fontweight='bold')
ax.set_ylabel('Returns', fontsize=12)

# Ajouter les statistiques
for i, (regime, returns) in enumerate(returns_by_regime.items()):
    mean = np.mean(returns)
    std = np.std(returns)
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)
    ax.text(i+1, ax.get_ylim()[1]*0.95, 
            f'μ={mean:.4f}\nσ={std:.4f}\nskew={skew:.2f}\nkurt={kurt:.2f}',
            ha='center', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.3))

# 2.2 Test de stationnarité
ax = axes[0, 1]
stationarity_results = []
for col in price_cols[:5]:
    # ADF test
    adf_result = adfuller(train[col].dropna())
    # KPSS test
    kpss_result = kpss(train[col].dropna(), regression='c')
    
    stationarity_results.append({
        'Asset': col[:10],
        'ADF Stat': adf_result[0],
        'ADF p-value': adf_result[1],
        'KPSS Stat': kpss_result[0],
        'KPSS p-value': kpss_result[1],
        'Stationnaire': 'Oui' if adf_result[1] < 0.05 and kpss_result[1] > 0.05 else 'Non'
    })

stat_df = pd.DataFrame(stationarity_results)
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=stat_df.round(4).values, 
                colLabels=stat_df.columns,
                cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)
ax.set_title('Tests de Stationnarité (ADF & KPSS)', fontsize=14, fontweight='bold', pad=20)

# 2.3 ANOVA - Différences entre régimes
ax = axes[1, 0]
# Préparer les données pour ANOVA
anova_data = []
for col in price_cols[:5]:
    returns = train[col].pct_change().dropna()
    regimes = train.loc[returns.index, 'Market_Regime']
    
    for ret, reg in zip(returns, regimes):
        anova_data.append({'Return': ret, 'Regime': reg, 'Asset': col})

anova_df = pd.DataFrame(anova_data)

# ANOVA pour chaque asset
f_stats = []
p_values = []
for asset in price_cols[:5]:
    asset_data = anova_df[anova_df['Asset'] == asset]
    groups = [asset_data[asset_data['Regime'] == r]['Return'].values for r in [-1, 0, 1]]
    f_stat, p_val = stats.f_oneway(*groups)
    f_stats.append(f_stat)
    p_values.append(p_val)

bars = ax.bar(range(len(f_stats)), f_stats, color='skyblue', alpha=0.7)
ax.set_title('ANOVA F-Statistics par Asset', fontsize=14, fontweight='bold')
ax.set_xlabel('Asset', fontsize=12)
ax.set_ylabel('F-Statistic', fontsize=12)
ax.set_xticks(range(len(f_stats)))
ax.set_xticklabels([f'Asset {i+1}' for i in range(len(f_stats))])

# Ajouter les p-values
for i, (bar, p_val) in enumerate(zip(bars, p_values)):
    height = bar.get_height()
    significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'p={p_val:.3f}\n{significance}', ha='center', va='bottom', fontsize=9)

# 2.4 Test de Tukey HSD
ax = axes[1, 1]
# Utiliser le premier asset pour le test
first_asset_data = anova_df[anova_df['Asset'] == price_cols[0]]
tukey = pairwise_tukeyhsd(endog=first_asset_data['Return'], 
                          groups=first_asset_data['Regime'], 
                          alpha=0.05)

# Visualiser les intervalles de confiance
tukey_summary = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
comparison_means = []
comparison_ci = []
comparison_names = []

for i, row in tukey_summary.iterrows():
    comparison_names.append(f"{row['group1']} vs {row['group2']}")
    comparison_means.append(float(row['meandiff']))
    comparison_ci.append([float(row['lower']), float(row['upper'])])

y_pos = np.arange(len(comparison_names))
means = np.array(comparison_means)
ci = np.array(comparison_ci)

ax.errorbar(means, y_pos, xerr=[means - ci[:, 0], ci[:, 1] - means], 
            fmt='o', color='darkblue', capsize=5, markersize=8)
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(comparison_names)
ax.set_title('Test de Tukey HSD - Comparaisons Multiples', fontsize=14, fontweight='bold')
ax.set_xlabel('Différence Moyenne', fontsize=12)
ax.grid(True, alpha=0.3)

# 2.5 Autocorrélation des régimes
ax = axes[2, 0]
from statsmodels.graphics.tsaplots import plot_acf
regime_series = train['Market_Regime']
plot_acf(regime_series, lags=40, ax=ax, alpha=0.05)
ax.set_title('Autocorrélation des Régimes de Marché', fontsize=14, fontweight='bold')
ax.set_xlabel('Lag', fontsize=12)
ax.set_ylabel('Autocorrélation', fontsize=12)

# 2.6 Distribution des transitions
ax = axes[2, 1]
transitions = []
for i in range(1, len(train)):
    prev_regime = train['Market_Regime'].iloc[i-1]
    curr_regime = train['Market_Regime'].iloc[i]
    transitions.append(f"{prev_regime}→{curr_regime}")

transition_counts = pd.Series(transitions).value_counts()
top_transitions = transition_counts.head(9)

bars = ax.bar(range(len(top_transitions)), top_transitions.values, color='coral', alpha=0.7)
ax.set_title('Fréquence des Transitions de Régime', fontsize=14, fontweight='bold')
ax.set_xlabel('Transition', fontsize=12)
ax.set_ylabel('Fréquence', fontsize=12)
ax.set_xticks(range(len(top_transitions)))
ax.set_xticklabels(top_transitions.index, rotation=45)

# Ajouter les pourcentages
total_transitions = len(transitions)
for bar, count in zip(bars, top_transitions.values):
    height = bar.get_height()
    pct = count / total_transitions * 100
    ax.text(bar.get_x() + bar.get_width()/2., height + 10,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('statistical_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. ANALYSE DE LA PERSISTANCE
print("\n[3] Analyse de la persistance des régimes...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 3.1 Matrice de Markov
ax = axes[0, 0]
# Calculer la matrice de transition
transition_matrix = pd.crosstab(train['Market_Regime'].shift(1), 
                               train['Market_Regime'], 
                               normalize='index')

sns.heatmap(transition_matrix, annot=True, fmt='.3f', cmap='YlOrRd', 
            ax=ax, cbar_kws={'label': 'Probabilité'})
ax.set_title('Matrice de Transition de Markov', fontsize=14, fontweight='bold')
ax.set_xlabel('Régime t+1', fontsize=12)
ax.set_ylabel('Régime t', fontsize=12)

# 3.2 Temps de séjour moyen
ax = axes[0, 1]
sojourn_times = {-1: [], 0: [], 1: []}
current_regime = train['Market_Regime'].iloc[0]
current_duration = 1

for i in range(1, len(train)):
    if train['Market_Regime'].iloc[i] == current_regime:
        current_duration += 1
    else:
        sojourn_times[current_regime].append(current_duration)
        current_regime = train['Market_Regime'].iloc[i]
        current_duration = 1

# Ajouter le dernier segment
sojourn_times[current_regime].append(current_duration)

# Calculer les statistiques
sojourn_stats = []
for regime in [-1, 0, 1]:
    times = sojourn_times[regime]
    sojourn_stats.append({
        'Regime': regime,
        'Moyenne': np.mean(times),
        'Médiane': np.median(times),
        'Max': np.max(times),
        'Min': np.min(times),
        'Std': np.std(times)
    })

sojourn_df = pd.DataFrame(sojourn_stats)
sojourn_df.set_index('Regime').plot(kind='bar', ax=ax)
ax.set_title('Statistiques du Temps de Séjour par Régime', fontsize=14, fontweight='bold')
ax.set_xlabel('Régime', fontsize=12)
ax.set_ylabel('Jours', fontsize=12)
ax.set_xticklabels(['Baissier (-1)', 'Neutre (0)', 'Haussier (1)'], rotation=0)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 3.3 Probabilité de changement au fil du temps
ax = axes[1, 0]
max_duration = 50
change_probs = {}
for regime in [-1, 0, 1]:
    probs = []
    for duration in range(1, max_duration + 1):
        # Compter combien de séquences ont duré exactement 'duration' jours
        # et combien ont duré au moins 'duration' jours
        exactly_duration = sum(1 for d in sojourn_times[regime] if d == duration)
        at_least_duration = sum(1 for d in sojourn_times[regime] if d >= duration)
        
        if at_least_duration > 0:
            prob_change = exactly_duration / at_least_duration
            probs.append(prob_change)
        else:
            probs.append(0)
    
    change_probs[regime] = probs

for regime, probs in change_probs.items():
    label = {-1: 'Baissier', 0: 'Neutre', 1: 'Haussier'}[regime]
    ax.plot(range(1, max_duration + 1), probs, linewidth=2, label=label, marker='o', markersize=4)

ax.set_title('Probabilité de Changement en Fonction de la Durée', fontsize=14, fontweight='bold')
ax.set_xlabel('Durée dans le Régime (jours)', fontsize=12)
ax.set_ylabel('Probabilité de Changement', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 30)

# 3.4 Analyse de survie
ax = axes[1, 1]
for regime in [-1, 0, 1]:
    sorted_durations = sorted(sojourn_times[regime], reverse=True)
    survival = np.arange(len(sorted_durations), 0, -1) / len(sorted_durations)
    label = {-1: 'Baissier', 0: 'Neutre', 1: 'Haussier'}[regime]
    ax.step(sorted_durations, survival, where='post', linewidth=2, label=label)

ax.set_title('Courbes de Survie des Régimes', fontsize=14, fontweight='bold')
ax.set_xlabel('Durée (jours)', fontsize=12)
ax.set_ylabel('Probabilité de Survie', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 50)

plt.tight_layout()
plt.savefig('persistence_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. ANALYSE SAISONNIÈRE ET CYCLIQUE
print("\n[4] Analyse saisonnière et cyclique...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 4.1 Heatmap par mois et jour de la semaine
ax = axes[0, 0]
train['DayOfWeek'] = train['Date'].dt.dayofweek
train['Month'] = train['Date'].dt.month
pivot_dow = pd.crosstab(train['DayOfWeek'], train['Market_Regime'], normalize='index')

days = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven']
sns.heatmap(pivot_dow.loc[:4], annot=True, fmt='.3f', cmap='RdYlGn', 
            ax=ax, yticklabels=days, xticklabels=['-1', '0', '1'])
ax.set_title('Distribution des Régimes par Jour de la Semaine', fontsize=14, fontweight='bold')
ax.set_xlabel('Régime', fontsize=12)
ax.set_ylabel('Jour', fontsize=12)

# 4.2 Patterns mensuels
ax = axes[0, 1]
monthly_regime_counts = pd.crosstab(train['Month'], train['Market_Regime'])
monthly_regime_pct = monthly_regime_counts.div(monthly_regime_counts.sum(axis=1), axis=0)

months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
for regime in [-1, 0, 1]:
    ax.plot(range(1, 13), monthly_regime_pct[regime], marker='o', linewidth=2, 
            label=f'Régime {regime}')

ax.set_title('Saisonnalité des Régimes', fontsize=14, fontweight='bold')
ax.set_xlabel('Mois', fontsize=12)
ax.set_ylabel('Proportion', fontsize=12)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(months, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)

# 4.3 Effet fin de mois
ax = axes[0, 2]
train['DayOfMonth'] = train['Date'].dt.day
train['IsMonthEnd'] = train['Date'].dt.is_month_end
train['IsMonthStart'] = train['Date'].dt.is_month_start

# Analyser les 5 premiers et derniers jours du mois
train['MonthPosition'] = 'Milieu'
train.loc[train['DayOfMonth'] <= 5, 'MonthPosition'] = 'Début'
train.loc[train['DayOfMonth'] >= 25, 'MonthPosition'] = 'Fin'

position_regime = pd.crosstab(train['MonthPosition'], train['Market_Regime'], normalize='index')
position_regime.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax.set_title('Effet de Position dans le Mois', fontsize=14, fontweight='bold')
ax.set_xlabel('Position dans le Mois', fontsize=12)
ax.set_ylabel('Proportion', fontsize=12)
ax.legend(['Baissier', 'Neutre', 'Haussier'], title='Régime')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# 4.4 Analyse par trimestre
ax = axes[1, 0]
train['Quarter'] = train['Date'].dt.quarter
quarterly_volatility = []
for quarter in range(1, 5):
    quarter_data = train[train['Quarter'] == quarter]
    vol = []
    for col in price_cols[:5]:
        returns = quarter_data[col].pct_change()
        vol.append(returns.std() * np.sqrt(252))
    quarterly_volatility.append(vol)

bp = ax.boxplot(quarterly_volatility, labels=['Q1', 'Q2', 'Q3', 'Q4'], patch_artist=True)
colors_quarterly = ['#3498db', '#2ecc71', '#f1c40f', '#e67e22']
for patch, color in zip(bp['boxes'], colors_quarterly):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_title('Volatilité par Trimestre', fontsize=14, fontweight='bold')
ax.set_xlabel('Trimestre', fontsize=12)
ax.set_ylabel('Volatilité Annualisée', fontsize=12)

# 4.5 Patterns annuels
ax = axes[1, 1]
train['Year'] = train['Date'].dt.year
yearly_regime_distribution = pd.crosstab(train['Year'], train['Market_Regime'], normalize='index')

yearly_regime_distribution.plot(ax=ax, marker='o', linewidth=2)
ax.set_title('Évolution Annuelle de la Distribution des Régimes', fontsize=14, fontweight='bold')
ax.set_xlabel('Année', fontsize=12)
ax.set_ylabel('Proportion', fontsize=12)
ax.legend(['Baissier', 'Neutre', 'Haussier'], title='Régime')
ax.grid(True, alpha=0.3)

# 4.6 Effet janvier et décembre
ax = axes[1, 2]
special_months = train[train['Month'].isin([1, 12])]
special_effect = pd.crosstab(special_months['Month'], special_months['Market_Regime'], normalize='index')
special_effect.index = ['Janvier', 'Décembre']

special_effect.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax.set_title('Effet Janvier vs Décembre', fontsize=14, fontweight='bold')
ax.set_ylabel('Proportion', fontsize=12)
ax.legend(['Baissier', 'Neutre', 'Haussier'], title='Régime')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# Ajouter une ligne de référence
avg_dist = train['Market_Regime'].value_counts(normalize=True).sort_index()
for i, regime in enumerate([-1, 0, 1]):
    ax.axhline(y=avg_dist[regime], color='black', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('seasonal_cyclical_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. RÉSUMÉ STATISTIQUE GLOBAL
print("\n[5] Création du résumé statistique global...")

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.axis('tight')
ax.axis('off')

# Créer un tableau récapitulatif
summary_stats = {
    'Métrique': [
        'Nombre total d\'observations',
        'Période couverte',
        'Nombre de changements de régime',
        'Fréquence de changement (%)',
        'Durée moyenne Baissier (jours)',
        'Durée moyenne Neutre (jours)',
        'Durée moyenne Haussier (jours)',
        'Autocorrélation lag-1',
        'Test ADF (stationnarité)',
        'Volatilité moyenne annualisée',
        'Return moyen annualisé',
        'Ratio de Sharpe estimé'
    ],
    'Valeur': [
        f"{len(train):,}",
        f"{train['Date'].min().strftime('%Y-%m-%d')} à {train['Date'].max().strftime('%Y-%m-%d')}",
        f"{(train['Market_Regime'] != train['Market_Regime'].shift(1)).sum():,}",
        f"{(train['Market_Regime'] != train['Market_Regime'].shift(1)).sum() / len(train) * 100:.1f}%",
        f"{np.mean(sojourn_times[-1]):.1f} ± {np.std(sojourn_times[-1]):.1f}",
        f"{np.mean(sojourn_times[0]):.1f} ± {np.std(sojourn_times[0]):.1f}",
        f"{np.mean(sojourn_times[1]):.1f} ± {np.std(sojourn_times[1]):.1f}",
        f"{train['Market_Regime'].autocorr(lag=1):.3f}",
        "Stationnaire" if all(adfuller(train[col].dropna())[1] < 0.05 for col in price_cols[:3]) else "Non-stationnaire",
        f"{np.mean([train[col].pct_change().std() * np.sqrt(252) for col in price_cols[:5]]):.3f}",
        f"{np.mean([train[col].pct_change().mean() * 252 for col in price_cols[:5]]):.3f}",
        f"{np.mean([train[col].pct_change().mean() * 252 / (train[col].pct_change().std() * np.sqrt(252)) for col in price_cols[:5]]):.2f}"
    ]
}

summary_df = pd.DataFrame(summary_stats)

# Créer le tableau
table = ax.table(cellText=summary_df.values,
                colLabels=summary_df.columns,
                cellLoc='left',
                loc='center',
                colWidths=[0.6, 0.4])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

# Style
for i in range(len(summary_df.columns)):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alterner les couleurs des lignes
for i in range(1, len(summary_df) + 1):
    if i % 2 == 0:
        for j in range(len(summary_df.columns)):
            table[(i, j)].set_facecolor('#f0f0f0')

ax.set_title('RÉSUMÉ STATISTIQUE GLOBAL', fontsize=18, fontweight='bold', pad=30)

plt.tight_layout()
plt.savefig('statistical_summary_table.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("ANALYSE STATISTIQUE COMPLÈTE TERMINÉE!")
print("\nFichiers générés:")
print("  - statistical_analysis.png : Tests statistiques et distributions")
print("  - persistence_analysis.png : Analyse de la persistance des régimes")
print("  - seasonal_cyclical_analysis.png : Patterns saisonniers et cycliques")
print("  - statistical_summary_table.png : Tableau récapitulatif global")
print("="*80)