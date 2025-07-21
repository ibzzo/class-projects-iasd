#!/usr/bin/env python3
"""
Script d'entraînement simple et efficace pour la détection de pathologies thoraciques
Version de base qui fonctionne bien
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = 'data'
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

print(f"Device: {DEVICE}")

class ChestXrayDataset(Dataset):
    """Dataset pour les radiographies thoraciques"""
    
    def __init__(self, annotations, img_dir, transforms=None):
        self.annotations = annotations
        self.img_dir = img_dir
        self.transforms = transforms
        
        # Grouper les annotations par image
        self.image_data = {}
        for idx, row in annotations.iterrows():
            img_name = row['image']
            if img_name not in self.image_data:
                self.image_data[img_name] = []
            
            self.image_data[img_name].append({
                'bbox': [row['x'], row['y'], row['x'] + row['w'], row['y'] + row['h']],
                'label': CLASSES.index(row['label']) + 1  # +1 car 0 est le background
            })
        
        self.images = list(self.image_data.keys())
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Charger l'image
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        
        # Obtenir les annotations
        annotations = self.image_data[img_name]
        
        # Convertir en tenseurs
        boxes = torch.as_tensor([ann['bbox'] for ann in annotations], dtype=torch.float32)
        labels = torch.as_tensor([ann['label'] for ann in annotations], dtype=torch.int64)
        
        # Créer le target
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        # Appliquer les transformations
        if self.transforms:
            img = self.transforms(img)
        
        return img, target

def get_model(num_classes):
    """Obtenir le modèle Faster R-CNN"""
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Remplacer le classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def train_one_epoch(model, data_loader, optimizer, device):
    """Entraîner pour une époque"""
    model.train()
    total_loss = 0
    
    for images, targets in tqdm(data_loader, desc="Training"):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    """Évaluer le modèle"""
    model.eval()
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Obtenir les prédictions
            predictions = model(images)
    
    return predictions

def generate_submission(model, test_dir, device):
    """Générer le fichier de soumission"""
    print("\nGénération de la soumission...")
    
    model.eval()
    
    # Charger le mapping
    id_mapping = pd.read_csv(os.path.join(DATA_DIR, 'ID_to_Image_Mapping.csv'))
    
    # Transformation pour les images de test
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    submission_data = []
    
    with torch.no_grad():
        for idx, row in tqdm(id_mapping.iterrows(), total=len(id_mapping)):
            img_id = idx + 1
            img_name = row['image_id']
            img_path = os.path.join(test_dir, img_name)
            
            if os.path.exists(img_path):
                # Charger et transformer l'image
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                # Prédiction
                predictions = model(img_tensor)[0]
                
                # Filtrer les prédictions avec score > 0.3
                if len(predictions['scores']) > 0:
                    # Prendre la meilleure prédiction
                    best_idx = predictions['scores'].argmax()
                    score = predictions['scores'][best_idx].item()
                    
                    if score > 0.3:
                        box = predictions['boxes'][best_idx].cpu().numpy()
                        label_idx = predictions['labels'][best_idx].item()
                        
                        submission_data.append({
                            'id': img_id,
                            'image_id': img_name,
                            'x_min': round(float(box[0]), 2),
                            'y_min': round(float(box[1]), 2),
                            'x_max': round(float(box[2]), 2),
                            'y_max': round(float(box[3]), 2),
                            'confidence': f"{score:.4f}",
                            'label': CLASSES[label_idx - 1] if label_idx > 0 else 'Cardiomegaly'
                        })
                    else:
                        # No Finding si score trop bas
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
                    # Pas de détection
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
    submission_df = pd.DataFrame(submission_data)
    submission_df = submission_df[['id', 'image_id', 'x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'label']]
    
    # Sauvegarder
    submission_df.to_csv('submission.csv', index=False)
    print(f"Soumission sauvegardée: {len(submission_df)} prédictions")
    
    # Afficher la distribution
    print("\nDistribution des prédictions:")
    print(submission_df['label'].value_counts())

def main():
    """Pipeline principal"""
    print("=== Entraînement Simple et Efficace ===\n")
    
    # Charger les données
    print("1. Chargement des données...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    train_df.columns = ['image', 'label', 'x', 'y', 'w', 'h']
    
    print(f"Total annotations: {len(train_df)}")
    print(f"Images uniques: {train_df['image'].nunique()}")
    
    # Split train/val
    unique_images = train_df['image'].unique()
    train_images, val_images = train_test_split(unique_images, test_size=0.2, random_state=42)
    
    train_annotations = train_df[train_df['image'].isin(train_images)]
    val_annotations = train_df[train_df['image'].isin(val_images)]
    
    print(f"\nTrain: {len(train_images)} images")
    print(f"Val: {len(val_images)} images")
    
    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Datasets
    train_dataset = ChestXrayDataset(
        train_annotations,
        os.path.join(DATA_DIR, 'train'),
        transforms=transform
    )
    
    val_dataset = ChestXrayDataset(
        val_annotations,
        os.path.join(DATA_DIR, 'train'),
        transforms=transform
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=0
    )
    
    # Modèle
    print("\n2. Initialisation du modèle...")
    model = get_model(num_classes=len(CLASSES) + 1)  # +1 pour le background
    model.to(DEVICE)
    
    # Optimiseur
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Entraînement
    print("\n3. Entraînement...")
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Sauvegarder si meilleur
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("✓ Modèle sauvegardé")
    
    # Charger le meilleur modèle
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Générer la soumission
    generate_submission(model, os.path.join(DATA_DIR, 'test'), DEVICE)
    
    print("\n✅ Entraînement terminé!")
    print("Fichiers créés:")
    print("- best_model.pth")
    print("- submission.csv")

if __name__ == "__main__":
    main()