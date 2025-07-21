# Guide d'Entraînement - Chest X-ray Detection

## 🎯 Objectif
Entraîner un modèle de détection d'objets pour identifier 8 pathologies thoraciques dans des radiographies.

## 📊 Classes à détecter
1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltrate
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax

## 🚀 Deux approches disponibles

### 1. YOLOv8 (Recommandé)
**Fichier**: `train_yolov8.py`

**Avantages**:
- Architecture moderne et performante
- Entraînement plus rapide
- Meilleure précision généralement
- Interface simple

**Installation**:
```bash
pip install ultralytics
```

**Utilisation**:
```bash
# Entraînement complet
python train_yolov8.py

# Générer uniquement la soumission avec un modèle existant
python train_yolov8.py --inference-only
```

**Paramètres importants**:
- `IMG_SIZE = 640` : Taille des images (peut être augmenté à 1024)
- `BATCH_SIZE = 8` : Taille du batch (réduire si mémoire insuffisante)
- `EPOCHS = 50` : Nombre d'époques
- `CONF_THRESHOLD = 0.15` : Seuil de confiance pour les prédictions

### 2. Faster R-CNN avec augmentations avancées
**Fichier**: `train_advanced.py`

**Avantages**:
- Plus de contrôle sur l'architecture
- Augmentations de données avancées
- Bon pour l'apprentissage

**Installation**:
```bash
pip install albumentations
```

**Utilisation**:
```bash
python train_advanced.py
```

## 📈 Améliorer les performances

### 1. Augmentation de données
- Rotation légère (±10°)
- Flip horizontal
- Ajustement de luminosité/contraste
- CLAHE (amélioration du contraste adaptatif)

### 2. Hyperparamètres
- Learning rate: Commencer à 0.001
- Scheduler: OneCycleLR ou CosineAnnealing
- Optimizer: AdamW ou SGD avec momentum

### 3. Post-traitement
- NMS (Non-Maximum Suppression) avec IoU = 0.4-0.5
- Seuil de confiance adaptatif par classe

### 4. Ensemble
Pour de meilleurs résultats, entraîner plusieurs modèles:
```python
# Modèle 1: YOLOv8s
# Modèle 2: YOLOv8m
# Modèle 3: Faster R-CNN

# Combiner les prédictions avec vote majoritaire ou moyenne
```

## 📝 Format de soumission

Le fichier `submission.csv` doit contenir:
```csv
id,image_id,x_min,y_min,x_max,y_max,confidence,label
1,00000865_006.png,343.34,428.51,876.03,826.53,0.5832,Cardiomegaly
```

**Important**:
- Si aucune détection: utiliser "No Finding" avec bbox [0,0,1,1]
- Confiance en format string avec 4 décimales
- Une seule détection par image (la plus confiante)

## 🔧 Conseils de débogage

1. **Vérifier les données**:
```python
# Visualiser quelques annotations
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def visualize_annotations(image_path, boxes, labels):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    for box, label in zip(boxes, labels):
        draw.rectangle(box, outline='red', width=3)
        draw.text((box[0], box[1]), label, fill='red')
    plt.imshow(img)
    plt.show()
```

2. **Monitor la loss**:
- La loss devrait diminuer progressivement
- Si elle stagne, réduire le learning rate
- Si elle explose, vérifier les annotations

3. **Validation**:
- Toujours garder 20% des données pour la validation
- Surveiller l'overfitting (train loss << val loss)

## 🏆 Stratégie pour Kaggle

1. **Commencer simple**: YOLOv8s avec paramètres par défaut
2. **Itérer rapidement**: Soumettre souvent pour voir les scores
3. **Analyser les erreurs**: Quelles classes sont mal détectées?
4. **Ajuster**: 
   - Augmenter les epochs si sous-apprentissage
   - Plus d'augmentations si surapprentissage
   - Ajuster les seuils de confiance par classe

## 📊 Métriques à surveiller

- **mAP@0.5**: Mean Average Precision (métrique principale)
- **Precision/Recall par classe**: Identifier les classes difficiles
- **Loss curves**: Pour détecter l'overfitting

## 🚨 Erreurs communes

1. **Mauvais format de bbox**: Vérifier [x,y,w,h] vs [x1,y1,x2,y2]
2. **Normalisation**: Les coordonnées doivent être en pixels, pas normalisées
3. **Classes manquantes**: S'assurer que toutes les classes sont dans le mapping
4. **Mémoire**: Réduire batch_size si "CUDA out of memory"

Bonne chance pour la compétition! 🎯