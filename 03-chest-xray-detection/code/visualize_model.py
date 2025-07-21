#!/usr/bin/env python3
"""
Visualisation des prédictions du modèle
Analyse et visualisation des résultats
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path('data')
OUTPUT_DIR = Path('visualizations')
OUTPUT_DIR.mkdir(exist_ok=True)

# Classes
CLASSES = [
    'Atelectasis',
    'Cardiomegaly', 
    'Effusion',
    'Infiltrate',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax'
]

# Couleurs pour chaque classe
COLORS = {
    'Atelectasis': '#FF6B6B',
    'Cardiomegaly': '#4ECDC4', 
    'Effusion': '#45B7D1',
    'Infiltrate': '#96CEB4',
    'Mass': '#FECA57',
    'Nodule': '#48DBFB',
    'Pneumonia': '#FF9FF3',
    'Pneumothorax': '#54A0FF',
    'No Finding': '#95A5A6'
}

print("=== Visualisation du Modèle ===\n")

def load_model(model_path='best_model.pth'):
    """Charger le modèle entraîné"""
    print(f"Chargement du modèle: {model_path}")
    
    # Créer le modèle
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASSES) + 1)
    
    # Charger les poids
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Modèle chargé (epoch {checkpoint.get('epoch', 'N/A')})")
        else:
            model.load_state_dict(checkpoint)
            print("✓ Modèle chargé")
    else:
        print("⚠️ Fichier modèle non trouvé!")
        return None
    
    model.eval()
    return model

def visualize_predictions_on_samples(model, n_samples=10):
    """Visualiser les prédictions sur des échantillons"""
    print("\n1. Visualisation des prédictions sur échantillons...")
    
    # Charger quelques images de test
    test_images = list((DATA_DIR / 'test').glob('*.png'))[:n_samples]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    transform = T.Compose([T.ToTensor()])
    
    for idx, img_path in enumerate(test_images):
        if idx >= 10:
            break
            
        ax = axes[idx]
        
        # Charger l'image
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        
        # Prédiction
        if model is not None:
            img_tensor = transform(img).unsqueeze(0)
            
            with torch.no_grad():
                predictions = model(img_tensor)[0]
            
            # Dessiner les boîtes
            for i in range(min(3, len(predictions['boxes']))):  # Max 3 boîtes
                if predictions['scores'][i] > 0.3:
                    box = predictions['boxes'][i].cpu().numpy()
                    label_idx = predictions['labels'][i].item()
                    score = predictions['scores'][i].item()
                    
                    if label_idx > 0 and label_idx <= len(CLASSES):
                        label = CLASSES[label_idx - 1]
                        color = COLORS.get(label, '#FF0000')
                        
                        # Convertir la couleur hex en RGB
                        color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                        
                        # Dessiner le rectangle
                        cv2.rectangle(img_array, 
                                    (int(box[0]), int(box[1])), 
                                    (int(box[2]), int(box[3])),
                                    color_rgb, 2)
                        
                        # Ajouter le label
                        cv2.putText(img_array, 
                                  f"{label} ({score:.2f})",
                                  (int(box[0]), int(box[1] - 5)),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, color_rgb, 2)
        
        ax.imshow(img_array)
        ax.set_title(f"{img_path.name}")
        ax.axis('off')
    
    plt.suptitle('Prédictions du Modèle sur Échantillons', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'predictions_samples.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Sauvegardé: predictions_samples.png")

def analyze_submission():
    """Analyser le fichier de soumission"""
    print("\n2. Analyse du fichier de soumission...")
    
    if not os.path.exists('submission.csv'):
        print("⚠️ Fichier submission.csv non trouvé!")
        return
    
    # Charger la soumission
    df = pd.read_csv('submission.csv')
    
    # 1. Distribution des classes
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Pie chart
    ax = axes[0, 0]
    class_counts = df['label'].value_counts()
    colors = [COLORS.get(label, '#333333') for label in class_counts.index]
    ax.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax.set_title('Distribution des Prédictions')
    
    # Bar chart
    ax = axes[0, 1]
    bars = ax.bar(class_counts.index, class_counts.values, color=colors)
    ax.set_xlabel('Pathologie')
    ax.set_ylabel('Nombre de prédictions')
    ax.set_title('Nombre de Prédictions par Classe')
    ax.tick_params(axis='x', rotation=45)
    
    # Ajouter les valeurs sur les barres
    for bar, count in zip(bars, class_counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')
    
    # 2. Distribution des scores de confiance
    ax = axes[1, 0]
    confidences = df[df['label'] != 'No Finding']['confidence'].astype(float)
    ax.hist(confidences, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Score de Confiance')
    ax.set_ylabel('Fréquence')
    ax.set_title('Distribution des Scores de Confiance')
    ax.axvline(confidences.mean(), color='red', linestyle='--', 
               label=f'Moyenne: {confidences.mean():.3f}')
    ax.legend()
    
    # 3. Scores moyens par classe
    ax = axes[1, 1]
    df_filtered = df[df['label'] != 'No Finding'].copy()
    df_filtered['confidence'] = df_filtered['confidence'].astype(float)
    
    mean_scores = df_filtered.groupby('label')['confidence'].mean().sort_values(ascending=False)
    colors_mean = [COLORS.get(label, '#333333') for label in mean_scores.index]
    
    bars = ax.bar(mean_scores.index, mean_scores.values, color=colors_mean)
    ax.set_xlabel('Pathologie')
    ax.set_ylabel('Score de Confiance Moyen')
    ax.set_title('Score de Confiance Moyen par Classe')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, 1)
    
    # Ajouter les valeurs
    for bar, score in zip(bars, mean_scores.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'submission_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Sauvegardé: submission_analysis.png")
    
    # 3. Heatmap des positions des détections
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Créer une heatmap des centres de boîtes
    heatmap = np.zeros((20, 20))
    
    for _, row in df[df['label'] != 'No Finding'].iterrows():
        x_center = (row['x_min'] + row['x_max']) / 2
        y_center = (row['y_min'] + row['y_max']) / 2
        
        # Normaliser en bins (supposant images 1024x1024)
        x_bin = min(int(x_center / 51.2), 19)
        y_bin = min(int(y_center / 51.2), 19)
        
        heatmap[y_bin, x_bin] += 1
    
    im = ax.imshow(heatmap, cmap='YlOrRd', aspect='auto')
    ax.set_title('Heatmap des Centres de Détection')
    ax.set_xlabel('Position X (bins)')
    ax.set_ylabel('Position Y (bins)')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'detection_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Sauvegardé: detection_heatmap.png")
    
    # 4. Statistiques détaillées
    print("\nStatistiques de la soumission:")
    print(f"Total prédictions: {len(df)}")
    print(f"No Finding: {(df['label'] == 'No Finding').sum()} ({(df['label'] == 'No Finding').mean()*100:.1f}%)")
    print(f"\nDistribution des classes:")
    print(class_counts)
    
    # Tailles moyennes des boîtes par classe
    df_boxes = df[df['label'] != 'No Finding'].copy()
    df_boxes['width'] = df_boxes['x_max'] - df_boxes['x_min']
    df_boxes['height'] = df_boxes['y_max'] - df_boxes['y_min']
    df_boxes['area'] = df_boxes['width'] * df_boxes['height']
    
    print("\nTailles moyennes des détections:")
    for label in df_boxes['label'].unique():
        mean_area = df_boxes[df_boxes['label'] == label]['area'].mean()
        print(f"  {label}: {mean_area:.0f} pixels²")

def visualize_confidence_distribution_by_class(df):
    """Visualiser la distribution des confidences par classe"""
    print("\n3. Distribution des confidences par classe...")
    
    df_filtered = df[df['label'] != 'No Finding'].copy()
    df_filtered['confidence'] = df_filtered['confidence'].astype(float)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Créer un violin plot
    classes = df_filtered['label'].unique()
    data_by_class = [df_filtered[df_filtered['label'] == cls]['confidence'].values 
                     for cls in classes]
    
    parts = ax.violinplot(data_by_class, positions=range(len(classes)), 
                          showmeans=True, showmedians=True)
    
    # Colorer les violons
    for pc, cls in zip(parts['bodies'], classes):
        pc.set_facecolor(COLORS.get(cls, '#333333'))
        pc.set_alpha(0.7)
    
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_xlabel('Pathologie')
    ax.set_ylabel('Score de Confiance')
    ax.set_title('Distribution des Scores de Confiance par Pathologie')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confidence_distribution_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Sauvegardé: confidence_distribution_by_class.png")

def compare_with_training_distribution():
    """Comparer avec la distribution d'entraînement"""
    print("\n4. Comparaison avec la distribution d'entraînement...")
    
    # Charger les données d'entraînement
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df.columns = ['image', 'label', 'x', 'y', 'w', 'h']
    
    # Charger la soumission
    submission_df = pd.read_csv('submission.csv')
    
    # Calculer les distributions
    train_dist = train_df['label'].value_counts(normalize=True).sort_index()
    
    # Pour la soumission, exclure No Finding
    submission_filtered = submission_df[submission_df['label'] != 'No Finding']
    submission_dist = submission_filtered['label'].value_counts(normalize=True).sort_index()
    
    # Créer le graphique de comparaison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(CLASSES))
    width = 0.35
    
    # Obtenir les valeurs pour chaque classe
    train_values = [train_dist.get(cls, 0) for cls in CLASSES]
    submission_values = [submission_dist.get(cls, 0) for cls in CLASSES]
    
    bars1 = ax.bar(x - width/2, train_values, width, label='Entraînement', 
                    color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, submission_values, width, label='Prédictions', 
                    color='lightcoral', alpha=0.8)
    
    # Ajouter les valeurs
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Pathologie')
    ax.set_ylabel('Proportion')
    ax.set_title('Comparaison des Distributions: Entraînement vs Prédictions')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Sauvegardé: distribution_comparison.png")

def create_summary_report():
    """Créer un rapport de synthèse"""
    print("\n5. Création du rapport de synthèse...")
    
    df = pd.read_csv('submission.csv')
    
    report = f"""
# Rapport de Visualisation du Modèle

## Résumé des Prédictions

- **Total de prédictions**: {len(df)}
- **Images avec détection**: {(df['label'] != 'No Finding').sum()}
- **Images sans détection (No Finding)**: {(df['label'] == 'No Finding').sum()}
- **Ratio No Finding**: {(df['label'] == 'No Finding').mean()*100:.1f}%

## Distribution des Classes Prédites

"""
    
    class_counts = df['label'].value_counts()
    for label, count in class_counts.items():
        report += f"- **{label}**: {count} ({count/len(df)*100:.1f}%)\n"
    
    # Scores de confiance
    df_filtered = df[df['label'] != 'No Finding'].copy()
    df_filtered['confidence'] = df_filtered['confidence'].astype(float)
    
    report += f"""

## Scores de Confiance

- **Moyenne globale**: {df_filtered['confidence'].mean():.3f}
- **Médiane**: {df_filtered['confidence'].median():.3f}
- **Min**: {df_filtered['confidence'].min():.3f}
- **Max**: {df_filtered['confidence'].max():.3f}

## Scores Moyens par Classe

"""
    
    mean_scores = df_filtered.groupby('label')['confidence'].mean().sort_values(ascending=False)
    for label, score in mean_scores.items():
        report += f"- **{label}**: {score:.3f}\n"
    
    report += """

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

"""
    
    with open(OUTPUT_DIR / 'rapport_visualisation.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✓ Rapport sauvegardé: rapport_visualisation.md")

def main():
    """Pipeline principal de visualisation"""
    
    # Vérifier l'existence de la soumission
    if not os.path.exists('submission.csv'):
        print("⚠️ Fichier submission.csv non trouvé!")
        print("Veuillez d'abord générer une soumission.")
        return
    
    # Charger le modèle si disponible
    model = None
    if os.path.exists('best_model.pth'):
        model = load_model('best_model.pth')
    elif os.path.exists('final_model.pth'):
        model = load_model('final_model.pth')
    
    # Visualisations
    if model is not None:
        visualize_predictions_on_samples(model)
    
    # Analyser la soumission
    df = pd.read_csv('submission.csv')
    analyze_submission()
    visualize_confidence_distribution_by_class(df)
    compare_with_training_distribution()
    
    # Créer le rapport
    create_summary_report()
    
    print("\n✅ Visualisation terminée!")
    print(f"Tous les fichiers sont dans le dossier '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()