
# Rapport de Visualisation du Modèle

## Résumé des Prédictions

- **Total de prédictions**: 148
- **Images avec détection**: 118
- **Images sans détection (No Finding)**: 30
- **Ratio No Finding**: 20.3%

## Distribution des Classes Prédites

- **No Finding**: 30 (20.3%)
- **Infiltrate**: 26 (17.6%)
- **Pneumonia**: 20 (13.5%)
- **Atelectasis**: 17 (11.5%)
- **Effusion**: 16 (10.8%)
- **Cardiomegaly**: 12 (8.1%)
- **Pneumothorax**: 10 (6.8%)
- **Mass**: 9 (6.1%)
- **Nodule**: 8 (5.4%)


## Scores de Confiance

- **Moyenne globale**: 0.449
- **Médiane**: 0.430
- **Min**: 0.085
- **Max**: 0.997

## Scores Moyens par Classe

- **Cardiomegaly**: 0.688
- **Infiltrate**: 0.563
- **Pneumonia**: 0.526
- **Nodule**: 0.446
- **Atelectasis**: 0.406
- **Mass**: 0.367
- **Pneumothorax**: 0.348
- **Effusion**: 0.148


## Visualisations Générées

1. **predictions_samples.png**: Exemples de prédictions sur images de test
2. **submission_analysis.png**: Analyse complète de la soumission
3. **detection_heatmap.png**: Heatmap des positions de détection
4. **confidence_distribution_by_class.png**: Distribution des confidences par classe
5. **distribution_comparison.png**: Comparaison avec la distribution d'entraînement

## Recommandations

1. **Équilibrage des classes**: Ajuster les seuils de confiance par classe
2. **Post-processing**: Appliquer des heuristiques médicales
3. **Augmentation**: Focus sur les classes sous-représentées
4. **Validation**: Vérifier la cohérence anatomique des détections

