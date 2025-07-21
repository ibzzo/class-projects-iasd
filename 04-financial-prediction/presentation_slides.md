# Prédiction de Séries Temporelles Financières
### Compétition Kaggle - Présentation des Résultats

---

## Slide 1: Introduction

### 📊 Contexte du Projet
- **Objectif** : Prédire des valeurs financières futures
- **Type** : Séries temporelles avec 133 features
- **Période** : 2008-2024 (15+ années de données)
- **Challenge** : Feature dominante à gérer

---

## Slide 2: Les Données

### 📈 Aperçu du Dataset

| Caractéristique | Train | Test |
|-----------------|-------|------|
| Observations | 2,811 | 1,206 |
| Période | 2008-2019 | 2019-2024 |
| Features | 133 | 133 |
| Target (moy.) | 0.136 | À prédire |

**Découverte clé** : Features_38 avec score d'importance = 1.0

---

## Slide 3: Méthodologie - Feature Engineering

### 🔧 Création de Features

1. **Temporelles** (20 features)
   - Date, cycliques (sin/cos), indicateurs

2. **Rolling Statistics** (300+ features)
   - Windows: 5, 7, 14, 21, 30, 60 jours
   - Mean, std, min, max, skew

3. **Lag Features** (200+ features)
   - Lags: 1, 2, 3, 5, 7, 14, 21, 30
   - Différences, ratios, % change

4. **Interactions** (100+ features)
   - Top features: ×, ÷, +, −
   - Polynomiales: x², x³, √x

**Total : 600-1100 features créées**

---

## Slide 4: Sélection des Features

### 🎯 Approche Multi-Critères

```
Score Final = 0.2×F-stat + 0.2×MI + 0.6×LightGBM
```

**Top 5 Features Sélectionnées** :
1. Features_101 (54.3%)
2. Features_41 (45.0%)
3. Features_104 (40.8%)
4. Features_44 (37.1%)
5. Features_50 (27.5%)

*Note: Features_38 exclue pour éviter l'overfitting*

---

## Slide 5: Stratégie de Modélisation

### 🤖 Ensemble de 6 Modèles

| Modèle | Hyperparamètres Clés |
|--------|---------------------|
| **LightGBM** | lr=0.008, leaves=40, depth=6 |
| **XGBoost** | lr=0.008, depth=6, γ=0.01 |
| **CatBoost** | lr=0.025, depth=7, l2=2 |
| **Random Forest** | trees=500, depth=10 |
| **Extra Trees** | trees=500, depth=10 |
| **Gradient Boost** | lr=0.008, depth=6 |

**Validation** : TimeSeriesSplit (7 folds)

---

## Slide 6: Résultats - Performance

### 📊 Scores de Validation Croisée

```
LightGBM    : 0.1244 ± 0.0193 ⭐
XGBoost     : 0.1266 ± 0.0204
CatBoost    : 0.1314 ± 0.0229
RF          : 0.1359 ± 0.0256
GB          : 0.1341 ± 0.0231
Extra Trees : 0.1573 ± 0.0254
```

**Poids Optimaux** :
- LightGBM : 98.6%
- CatBoost : 1.4%

---

## Slide 7: Post-Processing

### 🎨 Ajustements Finaux

1. **Valeurs négatives** → Correction si absent du train
2. **Outliers** → Clipping (percentiles 0.5-99.5)
3. **Variance** → Ajustement si ratio ≠ [0.5, 2.0]
4. **Lissage** → Moving average (window=3)

**Impact** :
- Stabilisation des prédictions
- Meilleure généralisation
- Réduction du bruit

---

## Slide 8: Features Importantes

### 🔍 Top 10 Contributeurs

| Rank | Feature | Type | Importance |
|------|---------|------|------------|
| 1 | Features_101 | Original | 3.07% |
| 2 | Features_41 | Original | 3.06% |
| 3 | Features_101_lag_1 | Lag | 2.31% |
| 4 | Features_101_roll_min_7 | Rolling | 2.14% |
| 5 | Features_44 | Original | 2.49% |

**Insights** :
- Features originales dominent
- Lag_1 très informatif
- Rolling sur 7 jours optimal

---

## Slide 9: Visualisations Clés

### 📈 Analyses Graphiques

1. **Distribution** : Match train/test ✓
2. **Évolution temporelle** : Tendances capturées ✓
3. **Q-Q Plot** : Normalité acceptable ✓
4. **Stabilité** : RMSE constant sur les folds ✓

![Performance par Fold]
- Fold 1-5: Stable (~0.12)
- Fold 6: Pic (0.16) - changement de régime?
- Fold 7: Retour normal (0.11)

---

## Slide 10: Version Fine-Tuned

### 🚀 Améliorations Avancées

**Changements** :
- 45 features (vs 40)
- 8 splits (vs 7)
- Plus de modèles (LightGBM v2)
- Features polynomiales

**Résultats** :
- Meilleur équilibre : LGB 43%, XGB 56%
- RMSE amélioré sur CV
- Plus robuste mais score Kaggle légèrement inférieur

---

## Slide 11: Leçons Apprises

### 💡 Insights Clés

**✅ Succès** :
1. Feature engineering extensif payant
2. LightGBM excellent pour séries temporelles
3. Validation temporelle stricte essentielle
4. Gestion proactive des features dominantes

**⚠️ Défis** :
1. Équilibrer complexité vs overfitting
2. Features_38 : blessing ou curse?
3. Ensemble déséquilibré mais performant

---

## Slide 12: Conclusion

### 🏆 Résultats Finaux

- **Meilleur Score** : Version Parameter Tuning
- **RMSE Final** : 0.1618
- **Approche Gagnante** : 
  - Feature engineering massif
  - LightGBM dominant (98.6%)
  - Post-processing minimal

### 📚 Contributions
- Pipeline ML complet et reproductible
- Gestion des séries temporelles sans data leakage
- Documentation détaillée des décisions

---

## Slide 13: Prochaines Étapes

### 🔮 Améliorations Futures

1. **Modèles Spécialisés**
   - LSTM/GRU pour patterns temporels
   - Prophet pour saisonnalité

2. **Feature Engineering**
   - Connaissance domaine finance
   - Indicateurs techniques

3. **Ensemble Avancé**
   - Stacking avec meta-learner
   - Blending dynamique

4. **Analyse Approfondie**
   - Étude des résidus
   - Détection d'anomalies

---

## Questions?

### 📧 Contact
- Code disponible sur GitHub
- Scripts Python modulaires
- Documentation complète

**Merci de votre attention!**

---

## Annexe: Commandes Clés

### Exécution du Meilleur Modèle

```bash
# Activation environnement
source venv_kaggle/bin/activate

# Lancer le modèle gagnant
python financial_prediction_parameter_tuning.py

# Output:
# ✅ submission_parameter_tuning.csv
# 📊 predictions_parameter_tuning_analysis.png
```

### Structure des Fichiers
```
kaggle_2/
├── data/
│   ├── train.csv
│   └── test.csv
├── financial_prediction_parameter_tuning.py ⭐
├── submission_parameter_tuning.csv
└── predictions_parameter_tuning_analysis.png
```