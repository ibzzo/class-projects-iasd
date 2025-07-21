#!/usr/bin/env python3
"""
Script YOLOv8 basique et fonctionnel
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

try:
    from ultralytics import YOLO
except ImportError:
    os.system("pip install ultralytics")
    from ultralytics import YOLO

# Configuration
DATA_DIR = Path("data")
DATASET_DIR = Path("yolo_dataset")
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 30

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

print("=== YOLOv8 Training ===")

def prepare_dataset():
    """Préparer le dataset au format YOLO"""
    print("\n1. Préparation du dataset...")
    
    # Créer les dossiers
    for split in ['train', 'val']:
        (DATASET_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Charger les annotations
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df.columns = ['image', 'label', 'x', 'y', 'w', 'h']
    
    print(f"Total annotations: {len(train_df)}")
    print(f"Images uniques: {train_df['image'].nunique()}")
    
    # Split train/val
    unique_images = train_df['image'].unique()
    train_images, val_images = train_test_split(unique_images, test_size=0.2, random_state=42)
    
    print(f"Train: {len(train_images)} images")
    print(f"Val: {len(val_images)} images")
    
    # Traiter les images
    for split, image_list in [('train', train_images), ('val', val_images)]:
        print(f"\nTraitement {split}...")
        
        for img_name in tqdm(image_list):
            # Copier l'image
            src = DATA_DIR / 'train' / img_name
            dst = DATASET_DIR / 'images' / split / img_name
            
            if src.exists():
                shutil.copy2(src, dst)
                
                # Créer le fichier label
                img_annotations = train_df[train_df['image'] == img_name]
                label_path = DATASET_DIR / 'labels' / split / f"{Path(img_name).stem}.txt"
                
                with open(label_path, 'w') as f:
                    for _, ann in img_annotations.iterrows():
                        # Index de la classe
                        class_idx = CLASSES.index(ann['label']) if ann['label'] in CLASSES else 0
                        
                        # Format YOLO : class x_center y_center width height (normalisé)
                        # Supposons des images 1024x1024
                        img_size = 1024
                        x_center = (ann['x'] + ann['w'] / 2) / img_size
                        y_center = (ann['y'] + ann['h'] / 2) / img_size
                        width = ann['w'] / img_size
                        height = ann['h'] / img_size
                        
                        # S'assurer que les valeurs sont valides
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))
                        
                        f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def create_yaml():
    """Créer le fichier YAML de configuration"""
    print("\n2. Création du fichier YAML...")
    
    config = {
        'path': str(DATASET_DIR.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(CLASSES),
        'names': CLASSES
    }
    
    yaml_path = DATASET_DIR / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    return yaml_path

def train_model(yaml_path):
    """Entraîner le modèle YOLOv8"""
    print("\n3. Entraînement du modèle...")
    
    # Charger le modèle
    model = YOLO('yolov8m.pt')  # medium model pour un bon compromis
    
    # Entraîner
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project='runs',
        name='chest_xray',
        patience=10,
        save=True,
        plots=True,
        conf=0.001,
        iou=0.5,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        box=7.5,
        cls=0.5,
        seed=42
    )
    
    print("\n✅ Entraînement terminé!")
    return model

def generate_submission(model):
    """Générer le fichier de soumission"""
    print("\n4. Génération de la soumission...")
    
    # Charger le mapping
    id_mapping = pd.read_csv(DATA_DIR / 'ID_to_Image_Mapping.csv')
    test_dir = DATA_DIR / 'test'
    
    submission_data = []
    
    for idx, row in tqdm(id_mapping.iterrows(), total=len(id_mapping)):
        img_id = idx + 1
        img_name = row['image_id']
        img_path = test_dir / img_name
        
        if img_path.exists():
            # Faire la prédiction
            results = model(img_path, imgsz=IMG_SIZE, conf=0.25, iou=0.45)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Prendre la meilleure détection
                boxes = results[0].boxes
                best_idx = boxes.conf.argmax()
                
                # Extraire les informations
                x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
                conf = float(boxes.conf[best_idx])
                cls = int(boxes.cls[best_idx])
                
                submission_data.append({
                    'id': img_id,
                    'image_id': img_name,
                    'x_min': round(float(x1), 2),
                    'y_min': round(float(y1), 2),
                    'x_max': round(float(x2), 2),
                    'y_max': round(float(y2), 2),
                    'confidence': f"{conf:.4f}",
                    'label': CLASSES[cls] if cls < len(CLASSES) else 'Cardiomegaly'
                })
            else:
                # Pas de détection - No Finding
                submission_data.append({
                    'id': img_id,
                    'image_id': img_name,
                    'x_min': 0.0,
                    'y_min': 0.0,
                    'x_max': 1.0,
                    'y_max': 1.0,
                    'confidence': "0.0000",
                    'label': 'No Finding'
                })
        else:
            # Image non trouvée
            submission_data.append({
                'id': img_id,
                'image_id': img_name,
                'x_min': 0.0,
                'y_min': 0.0,
                'x_max': 1.0,
                'y_max': 1.0,
                'confidence': "0.0000",
                'label': 'No Finding'
            })
    
    # Créer le DataFrame
    df = pd.DataFrame(submission_data)
    df = df[['id', 'image_id', 'x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'label']]
    
    # Sauvegarder
    df.to_csv('submission.csv', index=False)
    print(f"✅ Soumission sauvegardée: {len(df)} prédictions")
    
    # Afficher la distribution
    print("\nDistribution:")
    print(df['label'].value_counts())

def main():
    """Pipeline principal"""
    
    # Préparer le dataset
    prepare_dataset()
    
    # Créer le fichier YAML
    yaml_path = create_yaml()
    
    # Entraîner le modèle
    model = train_model(yaml_path)
    
    # Générer la soumission
    generate_submission(model)
    
    print("\n=== Terminé! ===")
    print("Fichiers créés:")
    print("- runs/chest_xray/weights/best.pt")
    print("- submission.csv")

if __name__ == "__main__":
    main()