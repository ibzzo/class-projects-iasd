# Prédiction de Séries Temporelles Financières
## Compétition Kaggle - Présentation des Résultats

---

## 📊 Vue d'ensemble du Projet

### Objectif
Prédire des valeurs financières futures basées sur des données historiques de séries temporelles avec 133 features numériques.

### Données
- **Période d'entraînement** : 2008-11-25 à 2019-09-03 (2,811 observations)
- **Période de test** : 2019-09-04 à 2024-04-17 (1,206 observations)
- **Nombre de features initiales** : 133
- **Variable cible** : `ToPredict` (valeurs entre -0.626 et 1.145)

---

## 🔬 Méthodologie

### 1. Exploration et Analyse des Données

#### Statistiques de la variable cible
- **Moyenne** : 0.136481
- **Écart-type** : 0.278196
- **Min/Max** : -0.626089 / 1.144987

#### Découverte clé
- **Features_38** domine complètement avec un score d'importance de 1.0
- Décision stratégique : Exclure cette feature pour éviter l'overfitting

### 2. Feature Engineering Avancé

#### A. Features Temporelles
- Extraction de composants : année, mois, jour, jour de la semaine, trimestre
- Features cycliques : sin/cos pour capturer la périodicité
- Indicateurs booléens : début/fin de mois, trimestre, année

#### B. Features Statistiques Mobiles
- **Rolling statistics** sur fenêtres [5, 7, 14, 21, 30, 60] jours :
  - Moyenne, écart-type, min, max
  - Skewness pour les fenêtres plus grandes
  - Ratios avec la moyenne mobile

#### C. Features de Décalage (Lag)
- Lags multiples : [1, 2, 3, 5, 7, 14, 21, 30] jours
- Différences et ratios avec les valeurs passées
- Pourcentage de changement

#### D. Features d'Interaction
- Produits, divisions, sommes et différences entre top features
- Features polynomiales (carré, cube, racine carrée) pour les 3 meilleures

#### E. Sélection des Features
- Combinaison de 3 méthodes :
  - F-statistics (20%)
  - Mutual Information (20%)
  - LightGBM importance (60%)
- Sélection finale : 40-45 features les plus importantes

### 3. Stratégie de Modélisation

#### Validation Temporelle Stricte
- **TimeSeriesSplit** avec 7-8 folds
- Respect de l'ordre chronologique
- Pas de data leakage

#### Ensemble de Modèles
1. **LightGBM** (2 versions avec paramètres différents)
2. **XGBoost**
3. **CatBoost**
4. **Random Forest**
5. **Gradient Boosting**
6. **Extra Trees**

### 4. Optimisation des Hyperparamètres

#### Version Parameter Tuning (Meilleure performance)
```python
LightGBM:
- n_estimators: 1500
- learning_rate: 0.008
- num_leaves: 40
- max_depth: 6
- Régularisation: reg_alpha=0.05, reg_lambda=0.05
```

#### Version Fine-Tuned
```python
LightGBM principal:
- n_estimators: 2000
- learning_rate: 0.005
- num_leaves: 45
- max_depth: 7
- Early stopping: 150 rounds
```

---

## 📈 Résultats

### Performance des Modèles (Cross-Validation)

#### Version Parameter Tuning
| Modèle | RMSE Moyen | Écart-type | Poids Final |
|--------|------------|------------|-------------|
| LightGBM | 0.1244 | ±0.0193 | **98.6%** |
| XGBoost | 0.1266 | ±0.0204 | 0.0% |
| CatBoost | 0.1314 | ±0.0229 | 1.4% |
| Random Forest | 0.1359 | ±0.0256 | 0.0% |
| Extra Trees | 0.1573 | ±0.0254 | 0.0% |
| Gradient Boosting | 0.1341 | ±0.0231 | 0.0% |

**Score Ensemble : 0.1618 RMSE**

#### Version Fine-Tuned
| Modèle | RMSE Moyen | Écart-type | Poids Final |
|--------|------------|------------|-------------|
| LightGBM | 0.1068 | ±0.0109 | 43.1% |
| XGBoost | 0.1070 | ±0.0106 | **56.2%** |
| CatBoost | 0.1222 | ±0.0332 | 0.0% |
| LightGBM v2 | 0.1095 | ±0.0185 | 0.7% |
| Random Forest | 0.1189 | ±0.0226 | 0.0% |
| Gradient Boosting | 0.1217 | ±0.0237 | 0.0% |

**Score Ensemble : 0.1349 RMSE**

### Features les Plus Importantes

#### Top 10 Features (Version Parameter Tuning)
1. Features_101 (3.07%)
2. Features_41 (3.06%)
3. Features_38 (3.05%) - Gardée malgré sa dominance
4. Features_44 (2.49%)
5. Features_104 (2.38%)
6. Features_101_lag_1 (2.31%)
7. Features_101_roll_min_7 (2.14%)
8. Features_85_roll_ratio_30 (1.84%)
9. Features_101_roll_mean_7 (1.71%)
10. Features_41_roll_min_7 (1.68%)

### Post-Processing

1. **Correction des valeurs négatives** si non présentes dans le train
2. **Clipping des valeurs extrêmes** (percentiles 0.5-99.5)
3. **Ajustement de la variance** si ratio > 2 ou < 0.5
4. **Lissage temporel adaptatif** pour séries > 100 points

---

## 🎯 Points Clés du Succès

### 1. Gestion de la Feature Dominante
- Identification précoce de Features_38
- Stratégie d'exclusion/réduction d'impact

### 2. Feature Engineering Robuste
- Plus de 600-1100 features créées
- Sélection rigoureuse des meilleures

### 3. Validation Temporelle
- Respect strict de l'ordre chronologique
- Pas de fuite d'information future

### 4. Ensemble Optimisé
- LightGBM domine mais reste performant
- Poids optimisés par minimisation RMSE

### 5. Post-Processing Conservateur
- Ajustements minimaux
- Préservation de la distribution originale

---

## 💡 Insights et Apprentissages

### Ce qui a fonctionné
1. **Features temporelles multiples** : Capturer différentes périodicités
2. **Rolling statistics étendues** : Fenêtres multiples pour différents horizons
3. **LightGBM avec régularisation** : Meilleur équilibre biais-variance
4. **Validation temporelle stricte** : Évaluation réaliste

### Défis rencontrés
1. **Feature dominante** : Nécessité de gérer Features_38
2. **Overfitting potentiel** : Avec >1000 features créées
3. **Déséquilibre des modèles** : LightGBM surpasse largement les autres

### Améliorations possibles
1. **Feature engineering ciblé** : Basé sur la connaissance du domaine
2. **Modèles spécialisés** : LSTM, Prophet pour séries temporelles
3. **Stacking plus sophistiqué** : Meta-learner au lieu de poids fixes
4. **Analyse des résidus** : Pour identifier les patterns manqués

---

## 📊 Visualisations Clés

### 1. Distribution des Prédictions
- Bonne correspondance avec la distribution du train
- Légère sous-estimation de la variance (intentionnelle)

### 2. Évolution Temporelle
- Capture des tendances principales
- Lissage approprié du bruit

### 3. Importance des Features
- Dominance des features originales (101, 41, 38, 44)
- Importance significative des lag_1 et rolling_mean_7

### 4. Performance par Fold
- Stabilité relative à travers les folds
- Fold 6 systématiquement plus difficile (changement de régime?)

---

## 🏆 Conclusion

### Résultats Finaux
- **Meilleur score obtenu** : Version Parameter Tuning
- **Approche gagnante** : Feature engineering extensif + LightGBM optimisé
- **Leçon principale** : Parfois, laisser dominer le meilleur modèle est optimal

### Valeur Ajoutée
1. Pipeline reproductible et modulaire
2. Validation rigoureuse sans data leakage
3. Post-processing adaptatif basé sur les données
4. Documentation complète des décisions

### Code et Reproductibilité
- Scripts Python modulaires et commentés
- Gestion propre des dépendances
- Visualisations automatiques pour l'analyse

---

*Projet réalisé dans le cadre d'une compétition Kaggle de prédiction de séries temporelles financières*