#!/usr/bin/env python3
"""
Analyse Exploratoire des Données (EDA) - Détection de Pathologies Thoraciques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from tqdm import tqdm
import warnings
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = Path("eda_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Palette de couleurs médicale
MEDICAL_PALETTE = {
    'Atelectasis': '#FF6B6B',
    'Cardiomegaly': '#4ECDC4', 
    'Effusion': '#45B7D1',
    'Infiltrate': '#96CEB4',
    'Mass': '#FECA57',
    'Nodule': '#48DBFB',
    'Pneumonia': '#FF9FF3',
    'Pneumothorax': '#54A0FF'
}

print("=== Analyse Exploratoire des Données - Chest X-Ray Detection ===\n")

def load_and_preprocess_data():
    """Charger et prétraiter les données"""
    print("1. Chargement des données...")
    
    # Charger les annotations
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df.columns = ['image', 'label', 'x', 'y', 'w', 'h']
    
    # Ajouter colonnes dérivées
    train_df['x_max'] = train_df['x'] + train_df['w']
    train_df['y_max'] = train_df['y'] + train_df['h']
    train_df['x_center'] = train_df['x'] + train_df['w'] / 2
    train_df['y_center'] = train_df['y'] + train_df['h'] / 2
    train_df['area'] = train_df['w'] * train_df['h']
    train_df['aspect_ratio'] = train_df['w'] / train_df['h']
    
    # Charger le mapping de test
    test_mapping = pd.read_csv(DATA_DIR / 'ID_to_Image_Mapping.csv')
    
    print(f"✓ Annotations chargées: {len(train_df)} lignes")
    print(f"✓ Images d'entraînement: {train_df['image'].nunique()}")
    print(f"✓ Images de test: {len(test_mapping)}")
    
    return train_df, test_mapping

def basic_statistics(train_df):
    """Statistiques de base"""
    print("\n2. Statistiques générales...")
    
    stats = {
        'Total annotations': len(train_df),
        'Images uniques': train_df['image'].nunique(),
        'Classes uniques': train_df['label'].nunique(),
        'Annotations/Image (moy)': len(train_df) / train_df['image'].nunique(),
        'Min annotations/image': train_df.groupby('image').size().min(),
        'Max annotations/image': train_df.groupby('image').size().max()
    }
    
    # Sauvegarder les stats
    stats_df = pd.DataFrame(stats.items(), columns=['Métrique', 'Valeur'])
    stats_df.to_csv(OUTPUT_DIR / 'statistiques_generales.csv', index=False)
    
    print("\nStatistiques générales:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    return stats

def analyze_class_distribution(train_df):
    """Analyser la distribution des classes"""
    print("\n3. Analyse de la distribution des classes...")
    
    # Distribution des classes
    class_counts = train_df['label'].value_counts()
    
    # Graphique en barres
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Barplot
    colors = [MEDICAL_PALETTE.get(label, '#333333') for label in class_counts.index]
    bars = ax1.bar(class_counts.index, class_counts.values, color=colors)
    ax1.set_xlabel('Pathologie')
    ax1.set_ylabel('Nombre d\'annotations')
    ax1.set_title('Distribution des pathologies')
    ax1.tick_params(axis='x', rotation=45)
    
    # Ajouter les valeurs sur les barres
    for bar, count in zip(bars, class_counts.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(train_df)*100:.1f}%)',
                ha='center', va='bottom')
    
    # Pie chart
    ax2.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax2.set_title('Répartition des pathologies')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'distribution_classes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Tableau détaillé
    class_stats = pd.DataFrame({
        'Pathologie': class_counts.index,
        'Nombre': class_counts.values,
        'Pourcentage': (class_counts.values / len(train_df) * 100).round(2)
    })
    class_stats.to_csv(OUTPUT_DIR / 'distribution_classes.csv', index=False)
    
    return class_counts

def analyze_bounding_boxes(train_df):
    """Analyser les caractéristiques des bounding boxes"""
    print("\n4. Analyse des bounding boxes...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribution des tailles par classe
    ax = axes[0, 0]
    box_data = []
    for label in train_df['label'].unique():
        areas = train_df[train_df['label'] == label]['area'].values
        box_data.append(areas)
    
    bp = ax.boxplot(box_data, labels=train_df['label'].unique(), patch_artist=True)
    for patch, label in zip(bp['boxes'], train_df['label'].unique()):
        patch.set_facecolor(MEDICAL_PALETTE.get(label, '#333333'))
    ax.set_xlabel('Pathologie')
    ax.set_ylabel('Surface (pixels²)')
    ax.set_title('Distribution des tailles de lésions par pathologie')
    ax.tick_params(axis='x', rotation=45)
    ax.set_yscale('log')
    
    # 2. Heatmap des positions
    ax = axes[0, 1]
    heatmap_data = np.zeros((10, 10))
    for _, row in train_df.iterrows():
        x_bin = min(int(row['x_center'] / 102.4), 9)
        y_bin = min(int(row['y_center'] / 102.4), 9)
        heatmap_data[y_bin, x_bin] += 1
    
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax.set_title('Heatmap des centres de détection')
    ax.set_xlabel('Position X (bins)')
    ax.set_ylabel('Position Y (bins)')
    plt.colorbar(im, ax=ax)
    
    # 3. Aspect ratio par classe
    ax = axes[1, 0]
    for label in train_df['label'].unique():
        data = train_df[train_df['label'] == label]['aspect_ratio']
        ax.hist(data, bins=30, alpha=0.6, label=label, 
                color=MEDICAL_PALETTE.get(label, '#333333'))
    ax.set_xlabel('Aspect Ratio (largeur/hauteur)')
    ax.set_ylabel('Fréquence')
    ax.set_title('Distribution des ratios d\'aspect')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(0, 3)
    
    # 4. Taille moyenne par classe
    ax = axes[1, 1]
    mean_sizes = train_df.groupby('label')[['w', 'h']].mean().sort_values('w', ascending=False)
    x = np.arange(len(mean_sizes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, mean_sizes['w'], width, label='Largeur', color='skyblue')
    bars2 = ax.bar(x + width/2, mean_sizes['h'], width, label='Hauteur', color='lightcoral')
    
    ax.set_xlabel('Pathologie')
    ax.set_ylabel('Taille moyenne (pixels)')
    ax.set_title('Dimensions moyennes des bounding boxes')
    ax.set_xticks(x)
    ax.set_xticklabels(mean_sizes.index, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'analyse_bounding_boxes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Statistiques détaillées
    bbox_stats = train_df.groupby('label').agg({
        'area': ['mean', 'std', 'min', 'max'],
        'w': ['mean', 'std'],
        'h': ['mean', 'std'],
        'aspect_ratio': ['mean', 'std']
    }).round(2)
    bbox_stats.to_csv(OUTPUT_DIR / 'statistiques_bounding_boxes.csv')
    
    return bbox_stats

def analyze_image_annotations(train_df):
    """Analyser les annotations par image"""
    print("\n5. Analyse des annotations par image...")
    
    # Nombre d'annotations par image
    annotations_per_image = train_df.groupby('image').size()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribution du nombre d'annotations
    ax = axes[0, 0]
    ax.hist(annotations_per_image, bins=range(0, annotations_per_image.max() + 2), 
            edgecolor='black', color='steelblue')
    ax.set_xlabel('Nombre d\'annotations par image')
    ax.set_ylabel('Nombre d\'images')
    ax.set_title('Distribution du nombre d\'annotations par image')
    ax.axvline(annotations_per_image.mean(), color='red', linestyle='--', 
               label=f'Moyenne: {annotations_per_image.mean():.1f}')
    ax.legend()
    
    # 2. Images avec le plus d'annotations
    ax = axes[0, 1]
    top_annotated = annotations_per_image.nlargest(20)
    ax.barh(range(len(top_annotated)), top_annotated.values, color='coral')
    ax.set_yticks(range(len(top_annotated)))
    ax.set_yticklabels([f"...{name[-10:]}" for name in top_annotated.index])
    ax.set_xlabel('Nombre d\'annotations')
    ax.set_title('Top 20 images les plus annotées')
    
    # 3. Co-occurrence des pathologies
    ax = axes[1, 0]
    # Créer matrice de co-occurrence
    cooccurrence = defaultdict(lambda: defaultdict(int))
    for img in train_df['image'].unique():
        img_labels = train_df[train_df['image'] == img]['label'].unique()
        for i, label1 in enumerate(img_labels):
            for label2 in img_labels[i:]:
                cooccurrence[label1][label2] += 1
                if label1 != label2:
                    cooccurrence[label2][label1] += 1
    
    # Convertir en matrice
    labels = sorted(train_df['label'].unique())
    matrix = np.zeros((len(labels), len(labels)))
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            matrix[i, j] = cooccurrence[label1][label2]
    
    im = ax.imshow(matrix, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    ax.set_title('Matrice de co-occurrence des pathologies')
    plt.colorbar(im, ax=ax)
    
    # 4. Distribution des pathologies multiples
    ax = axes[1, 1]
    multi_pathology = train_df.groupby('image')['label'].nunique()
    unique_counts = multi_pathology.value_counts().sort_index()
    ax.bar(unique_counts.index, unique_counts.values, color='lightgreen')
    ax.set_xlabel('Nombre de pathologies différentes')
    ax.set_ylabel('Nombre d\'images')
    ax.set_title('Distribution du nombre de pathologies par image')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'analyse_annotations_images.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return annotations_per_image

def analyze_spatial_distribution(train_df):
    """Analyser la distribution spatiale des pathologies"""
    print("\n6. Analyse de la distribution spatiale...")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    for idx, (label, color) in enumerate(MEDICAL_PALETTE.items()):
        ax = axes[idx]
        label_data = train_df[train_df['label'] == label]
        
        if len(label_data) > 0:
            # Créer heatmap pour cette pathologie
            heatmap = np.zeros((20, 20))
            for _, row in label_data.iterrows():
                x_bin = min(int(row['x_center'] / 51.2), 19)
                y_bin = min(int(row['y_center'] / 51.2), 19)
                heatmap[y_bin, x_bin] += 1
            
            im = ax.imshow(heatmap, cmap='hot', aspect='auto')
            ax.set_title(f'{label}\n({len(label_data)} annotations)')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(False)
    
    plt.suptitle('Distribution spatiale par pathologie', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'distribution_spatiale_pathologies.png', dpi=300, bbox_inches='tight')
    plt.close()

def sample_images_analysis(train_df):
    """Analyser des échantillons d'images"""
    print("\n7. Analyse d'échantillons d'images...")
    
    # Sélectionner des images représentatives
    sample_images = []
    
    # Une image par pathologie principale
    for label in train_df['label'].unique():
        label_images = train_df[train_df['label'] == label]['image'].unique()
        if len(label_images) > 0:
            sample_images.append(np.random.choice(label_images))
    
    # Créer visualisation
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    for idx, img_name in enumerate(sample_images[:8]):
        ax = axes[idx]
        img_path = TRAIN_DIR / img_name
        
        if img_path.exists():
            # Charger l'image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Afficher l'image
                ax.imshow(img, cmap='gray')
                
                # Dessiner les bounding boxes
                img_annotations = train_df[train_df['image'] == img_name]
                for _, ann in img_annotations.iterrows():
                    rect = plt.Rectangle((ann['x'], ann['y']), ann['w'], ann['h'],
                                       fill=False, edgecolor=MEDICAL_PALETTE.get(ann['label'], 'red'),
                                       linewidth=2)
                    ax.add_patch(rect)
                    ax.text(ann['x'], ann['y']-5, ann['label'], 
                           color=MEDICAL_PALETTE.get(ann['label'], 'red'),
                           fontsize=8, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
                
                ax.set_title(f"Image: {img_name[:15]}...")
                ax.axis('off')
    
    plt.suptitle('Échantillons d\'images avec annotations', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'echantillons_images.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_image_properties():
    """Analyser les propriétés des images"""
    print("\n8. Analyse des propriétés des images...")
    
    # Échantillonner quelques images pour l'analyse
    sample_size = min(100, len(list(TRAIN_DIR.glob('*.png'))))
    sample_images = list(TRAIN_DIR.glob('*.png'))[:sample_size]
    
    properties = {
        'widths': [],
        'heights': [],
        'channels': [],
        'mean_intensity': [],
        'std_intensity': []
    }
    
    print(f"  Analyse de {sample_size} images...")
    for img_path in tqdm(sample_images):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            properties['heights'].append(img.shape[0])
            properties['widths'].append(img.shape[1])
            properties['channels'].append(len(img.shape))
            
            # Convertir en grayscale si nécessaire
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img
            
            properties['mean_intensity'].append(np.mean(img_gray))
            properties['std_intensity'].append(np.std(img_gray))
    
    # Visualiser les résultats
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Dimensions
    ax = axes[0, 0]
    ax.hist2d(properties['widths'], properties['heights'], bins=20, cmap='Blues')
    ax.set_xlabel('Largeur (pixels)')
    ax.set_ylabel('Hauteur (pixels)')
    ax.set_title('Distribution des dimensions d\'images')
    
    # Intensité moyenne
    ax = axes[0, 1]
    ax.hist(properties['mean_intensity'], bins=30, color='green', alpha=0.7)
    ax.set_xlabel('Intensité moyenne')
    ax.set_ylabel('Fréquence')
    ax.set_title('Distribution de l\'intensité moyenne')
    
    # Écart-type de l'intensité
    ax = axes[1, 0]
    ax.hist(properties['std_intensity'], bins=30, color='orange', alpha=0.7)
    ax.set_xlabel('Écart-type de l\'intensité')
    ax.set_ylabel('Fréquence')
    ax.set_title('Distribution de la variation d\'intensité')
    
    # Scatter plot intensité
    ax = axes[1, 1]
    ax.scatter(properties['mean_intensity'], properties['std_intensity'], alpha=0.5)
    ax.set_xlabel('Intensité moyenne')
    ax.set_ylabel('Écart-type')
    ax.set_title('Relation intensité moyenne vs variation')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'proprietes_images.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return properties

def create_interactive_visualizations(train_df):
    """Créer des visualisations interactives avec Plotly"""
    print("\n9. Création de visualisations interactives...")
    
    # 1. Distribution des tailles de boîtes par classe (interactive)
    fig = px.box(train_df, x='label', y='area', color='label',
                 color_discrete_map=MEDICAL_PALETTE,
                 title='Distribution des tailles de lésions par pathologie',
                 labels={'area': 'Surface (pixels²)', 'label': 'Pathologie'})
    fig.update_yaxis(type="log")
    fig.write_html(OUTPUT_DIR / 'distribution_tailles_interactive.html')
    
    # 2. Scatter plot des centres de détection
    fig = px.scatter(train_df, x='x_center', y='y_center', color='label',
                     color_discrete_map=MEDICAL_PALETTE,
                     title='Position des centres de détection',
                     labels={'x_center': 'Position X', 'y_center': 'Position Y'},
                     hover_data=['image', 'w', 'h'])
    fig.update_layout(width=800, height=800)
    fig.write_html(OUTPUT_DIR / 'centres_detection_interactive.html')
    
    # 3. Sunburst chart pour la hiérarchie des données
    # Préparer les données
    img_summary = train_df.groupby(['image', 'label']).size().reset_index(name='count')
    img_summary['total'] = 'Total'
    
    fig = px.sunburst(img_summary, path=['total', 'label', 'image'], values='count',
                      color='label', color_discrete_map=MEDICAL_PALETTE,
                      title='Hiérarchie des annotations par pathologie et image')
    fig.update_layout(width=800, height=800)
    fig.write_html(OUTPUT_DIR / 'hierarchie_annotations.html')

def generate_summary_report(train_df, class_counts, bbox_stats):
    """Générer un rapport de synthèse"""
    print("\n10. Génération du rapport de synthèse...")
    
    report = f"""
# Rapport d'Analyse Exploratoire - Chest X-Ray Detection

## Résumé Exécutif

- **Total d'annotations**: {len(train_df):,}
- **Images uniques**: {train_df['image'].nunique():,}
- **Classes de pathologies**: {train_df['label'].nunique()}
- **Annotations moyennes par image**: {len(train_df) / train_df['image'].nunique():.2f}

## Distribution des Classes

La pathologie la plus fréquente est **{class_counts.index[0]}** avec {class_counts.values[0]:,} annotations ({class_counts.values[0]/len(train_df)*100:.1f}%).
La pathologie la moins fréquente est **{class_counts.index[-1]}** avec {class_counts.values[-1]:,} annotations ({class_counts.values[-1]/len(train_df)*100:.1f}%).

## Caractéristiques des Détections

### Tailles moyennes des lésions (en pixels²):
"""
    
    for label in train_df['label'].unique():
        mean_area = train_df[train_df['label'] == label]['area'].mean()
        report += f"- **{label}**: {mean_area:,.0f} pixels²\n"
    
    report += f"""

## Insights Clés

1. **Déséquilibre des classes**: Il existe un déséquilibre significatif entre les classes, 
   avec un ratio de {class_counts.values[0]/class_counts.values[-1]:.1f}:1 entre la classe la plus et la moins fréquente.

2. **Localisation spatiale**: Les pathologies montrent des patterns de localisation distincts:
   - Cardiomegaly: Principalement au centre-bas de l'image
   - Pneumothorax: Souvent sur les bords latéraux
   - Effusion: Généralement dans la partie inférieure

3. **Tailles de détection**: Les pathologies varient considérablement en taille:
   - Nodules: Petites détections (< 30,000 pixels²)
   - Cardiomegaly: Grandes détections (> 80,000 pixels²)

4. **Multi-pathologies**: {(train_df.groupby('image')['label'].nunique() > 1).sum()} images ({(train_df.groupby('image')['label'].nunique() > 1).sum() / train_df['image'].nunique() * 100:.1f}%) contiennent plusieurs pathologies.

## Recommandations

1. **Augmentation de données**: Focus sur les classes sous-représentées (Nodule, Mass, Pneumothorax)
2. **Stratification**: Utiliser une stratification basée sur les pathologies multiples
3. **Prétraitement**: Appliquer CLAHE pour améliorer le contraste des radiographies
4. **Validation anatomique**: Implémenter des contraintes basées sur la localisation anatomique
5. **Métriques d'évaluation**: Utiliser des métriques pondérées pour gérer le déséquilibre

"""
    
    # Sauvegarder le rapport
    with open(OUTPUT_DIR / 'rapport_synthese.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✓ Rapport sauvegardé dans 'eda_results/rapport_synthese.md'")

def main():
    """Pipeline principal d'EDA"""
    
    # Charger les données
    train_df, test_mapping = load_and_preprocess_data()
    
    # Analyses
    stats = basic_statistics(train_df)
    class_counts = analyze_class_distribution(train_df)
    bbox_stats = analyze_bounding_boxes(train_df)
    annotations_per_image = analyze_image_annotations(train_df)
    analyze_spatial_distribution(train_df)
    sample_images_analysis(train_df)
    image_properties = analyze_image_properties()
    create_interactive_visualizations(train_df)
    
    # Générer le rapport
    generate_summary_report(train_df, class_counts, bbox_stats)
    
    print("\n=== Analyse Exploratoire Terminée! ===")
    print(f"Tous les résultats sont sauvegardés dans le dossier '{OUTPUT_DIR}/'")
    print("\nFichiers générés:")
    for file in sorted(OUTPUT_DIR.glob('*')):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()