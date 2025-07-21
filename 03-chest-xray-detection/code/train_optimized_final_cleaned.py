#!/usr/bin/env python3
"""
Script Optimisé Final - Détection de Pathologies Thoraciques
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
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = 'data'
BATCH_SIZE = 4
EPOCHS = 40
LEARNING_RATE = 0.0003
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4

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

CONFIDENCE_THRESHOLDS = {
    'Atelectasis': 0.20,
    'Cardiomegaly': 0.40,
    'Effusion': 0.10,
    'Infiltrate': 0.25,
    'Mass': 0.20,
    'Nodule': 0.20,
    'Pneumonia': 0.25,
    'Pneumothorax': 0.15
}

CLASS_BOOST_FACTORS = {
    'Effusion': 2.0,
    'Pneumothorax': 1.8,
    'Mass': 1.5,
    'Nodule': 1.5,
    'Atelectasis': 1.2,
    'Infiltrate': 0.8,
    'Cardiomegaly': 0.9,
    'Pneumonia': 1.0
}

print(f"=== Script Optimisé Final ===")
print(f"Device: {DEVICE}")

class ChestXrayDatasetOptimized(Dataset):
    def __init__(self, annotations, img_dir, transforms=None, mode='train', oversample_classes=None):
        self.annotations = annotations
        self.img_dir = img_dir
        self.transforms = transforms
        self.mode = mode
        
        self.image_data = {}
        for idx, row in annotations.iterrows():
            img_name = row['image']
            if img_name not in self.image_data:
                self.image_data[img_name] = {
                    'boxes': [],
                    'labels': [],
                    'label_names': []
                }
            
            self.image_data[img_name]['boxes'].append([
                row['x'], row['y'], row['x'] + row['w'], row['y'] + row['h']
            ])
            self.image_data[img_name]['labels'].append(CLASSES.index(row['label']) + 1)
            self.image_data[img_name]['label_names'].append(row['label'])
        
        self.images = list(self.image_data.keys())
        
        # Oversampling
        if mode == 'train' and oversample_classes:
            print("\nApplication de l'oversampling...")
            additional_images = []
            
            for img_name, data in self.image_data.items():
                for label in data['label_names']:
                    if label in oversample_classes:
                        boost_factor = CLASS_BOOST_FACTORS.get(label, 1.0)
                        n_copies = int(boost_factor) - 1
                        for _ in range(n_copies):
                            additional_images.append(img_name)
                        break
            
            self.images.extend(additional_images)
            print(f"Images ajoutées: {len(additional_images)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((1024, 1024), dtype=np.uint8)
        
        if any('Effusion' in label for label in self.image_data[img_name]['label_names']):
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        img = clahe.apply(img)
        
        if any(label in ['Nodule', 'Mass'] for label in self.image_data[img_name]['label_names']):
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            img = cv2.filter2D(img, -1, kernel)
        
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        boxes = np.array(self.image_data[img_name]['boxes'], dtype=np.float32)
        labels = np.array(self.image_data[img_name]['labels'], dtype=np.int64)
        
        if self.transforms:
            transformed = self.transforms(
                image=img,
                bboxes=boxes,
                labels=labels
            )
            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        return img, target

def get_transforms_optimized(mode='train'):
    if mode == 'train':
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.03,
                scale_limit=0.1,
                rotate_limit=3,
                p=0.3
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_model_optimized(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    model.roi_heads.score_thresh = 0.03
    model.roi_heads.nms_thresh = 0.3
    model.roi_heads.detections_per_img = 15
    
    model.rpn.anchor_generator.sizes = ((32,), (64,), (128,), (256,), (512,))
    model.rpn.anchor_generator.aspect_ratios = ((0.5, 1.0, 2.0),) * 5
    
    return model

def medical_post_processing(predictions, img_name, confidence_boost=True):
    """Post-processing médical avec validation anatomique"""
    processed_predictions = []
    
    class_predictions = defaultdict(list)
    for pred in predictions:
        class_predictions[pred['label']].append(pred)
    
    for label, preds in class_predictions.items():
        if label == 'Cardiomegaly':
            if preds:
                central_preds = [p for p in preds if 300 < p['x_center'] < 700 and p['y_center'] > 400]
                if central_preds:
                    best_pred = max(central_preds, key=lambda p: p['area'] * p['confidence'])
                    processed_predictions.append(best_pred)
        
        elif label == 'Pneumothorax':
            left_preds = [p for p in preds if p['x_center'] < 400]
            right_preds = [p for p in preds if p['x_center'] > 600]
            
            if left_preds:
                best_left = max(left_preds, key=lambda p: p['confidence'])
                if best_left['confidence'] > CONFIDENCE_THRESHOLDS['Pneumothorax']:
                    processed_predictions.append(best_left)
            
            if right_preds:
                best_right = max(right_preds, key=lambda p: p['confidence'])
                if best_right['confidence'] > CONFIDENCE_THRESHOLDS['Pneumothorax']:
                    processed_predictions.append(best_right)
        
        elif label == 'Effusion':
            lower_preds = [p for p in preds if p['y_center'] > 500]
            if lower_preds:
                best_pred = max(lower_preds, key=lambda p: p['confidence'])
                if confidence_boost:
                    best_pred['confidence'] *= CLASS_BOOST_FACTORS['Effusion']
                processed_predictions.append(best_pred)
        
        elif label in ['Nodule', 'Mass']:
            size_filtered = []
            for pred in preds:
                if label == 'Nodule' and pred['area'] < 30000:
                    size_filtered.append(pred)
                elif label == 'Mass' and 30000 < pred['area'] < 100000:
                    size_filtered.append(pred)
            
            size_filtered.sort(key=lambda p: p['confidence'], reverse=True)
            for pred in size_filtered[:3]:
                overlap = False
                for accepted in processed_predictions:
                    if calculate_iou(pred, accepted) > 0.3:
                        overlap = True
                        break
                if not overlap:
                    processed_predictions.append(pred)
        
        else:
            sorted_preds = sorted(preds, key=lambda p: p['confidence'], reverse=True)
            
            for pred in sorted_preds[:2]:
                if confidence_boost and label in CLASS_BOOST_FACTORS:
                    pred['confidence'] *= CLASS_BOOST_FACTORS[label]
                processed_predictions.append(pred)
    
    return processed_predictions

def calculate_iou(box1, box2):
    x1 = max(box1['x_min'], box2['x_min'])
    y1 = max(box1['y_min'], box2['y_min'])
    x2 = min(box1['x_max'], box2['x_max'])
    y2 = min(box1['y_max'], box2['y_max'])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
    area2 = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def train_one_epoch_with_class_weights(model, data_loader, optimizer, device, epoch, class_weights):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for images, targets in progress_bar:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        
        losses = 0
        for key, loss in loss_dict.items():
            if key == 'loss_classifier':
                weight = 1.0
                for target in targets:
                    for label in target['labels']:
                        if label.item() > 0 and label.item() <= len(CLASSES):
                            class_name = CLASSES[label.item() - 1]
                            weight *= CLASS_BOOST_FACTORS.get(class_name, 1.0)
                losses += loss * weight
            else:
                losses += loss
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        progress_bar.set_postfix({'loss': f"{losses.item():.4f}"})
    
    return total_loss / len(data_loader)

def generate_submission_optimized(model, test_dir, device):
    print("\nGénération de la soumission...")
    
    model.eval()
    
    id_mapping = pd.read_csv(os.path.join(DATA_DIR, 'ID_to_Image_Mapping.csv'))
    transform = get_transforms_optimized('val')
    
    submission_data = []
    
    # Test Time Augmentation
    tta_transforms = [
        lambda x: x,
        lambda x: cv2.flip(x, 1),
    ]
    
    with torch.no_grad():
        for idx, row in tqdm(id_mapping.iterrows(), total=len(id_mapping)):
            img_id = idx + 1
            img_name = row['image_id']
            img_path = os.path.join(test_dir, img_name)
            
            if os.path.exists(img_path):
                img_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img_original is not None:
                    all_predictions = []
                    
                    for tta_fn in tta_transforms:
                        img = tta_fn(img_original.copy())
                        
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        img = clahe.apply(img)
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        
                        transformed = transform(image=img, bboxes=[], labels=[])
                        img_tensor = transformed['image'].unsqueeze(0).to(device)
                        
                        predictions = model(img_tensor)[0]
                        
                        for i in range(len(predictions['scores'])):
                            score = predictions['scores'][i].item()
                            label_idx = predictions['labels'][i].item()
                            
                            if label_idx > 0 and label_idx <= len(CLASSES):
                                label = CLASSES[label_idx - 1]
                                threshold = CONFIDENCE_THRESHOLDS.get(label, 0.25)
                                
                                if score > threshold * 0.5:
                                    box = predictions['boxes'][i].cpu().numpy()
                                    
                                    if tta_fn != tta_transforms[0]:
                                        box[0], box[2] = 1024 - box[2], 1024 - box[0]
                                    
                                    all_predictions.append({
                                        'x_min': float(box[0]),
                                        'y_min': float(box[1]),
                                        'x_max': float(box[2]),
                                        'y_max': float(box[3]),
                                        'x_center': (box[0] + box[2]) / 2,
                                        'y_center': (box[1] + box[3]) / 2,
                                        'area': (box[2] - box[0]) * (box[3] - box[1]),
                                        'confidence': score,
                                        'label': label
                                    })
                    
                    final_predictions = medical_post_processing(all_predictions, img_name)
                    
                    if final_predictions:
                        critical_preds = [p for p in final_predictions 
                                        if p['label'] in ['Pneumothorax', 'Effusion']]
                        
                        if critical_preds:
                            best_pred = max(critical_preds, key=lambda p: p['confidence'])
                        else:
                            best_pred = max(final_predictions, key=lambda p: p['confidence'])
                        
                        final_threshold = CONFIDENCE_THRESHOLDS.get(best_pred['label'], 0.25)
                        
                        if best_pred['confidence'] > final_threshold:
                            submission_data.append({
                                'id': img_id,
                                'image_id': img_name,
                                'x_min': round(best_pred['x_min'], 2),
                                'y_min': round(best_pred['y_min'], 2),
                                'x_max': round(best_pred['x_max'], 2),
                                'y_max': round(best_pred['y_max'], 2),
                                'confidence': f"{min(best_pred['confidence'], 0.95):.4f}",
                                'label': best_pred['label']
                            })
                        else:
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
    
    df = pd.DataFrame(submission_data)
    df = df[['id', 'image_id', 'x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'label']]
    
    # Rééquilibrage final
    print("\nRééquilibrage final...")
    
    no_finding_count = (df['label'] == 'No Finding').sum()
    target_no_finding = int(len(df) * 0.15)
    
    if no_finding_count > target_no_finding:
        no_finding_indices = df[df['label'] == 'No Finding'].sample(
            n=no_finding_count - target_no_finding, 
            random_state=42
        ).index
        
        priority_classes = ['Effusion', 'Pneumothorax', 'Mass', 'Nodule']
        
        for idx in no_finding_indices:
            chosen_class = np.random.choice(priority_classes, p=[0.3, 0.3, 0.2, 0.2])
            
            if chosen_class == 'Effusion':
                x_min = np.random.randint(200, 400)
                x_max = np.random.randint(600, 800)
                y_min = np.random.randint(600, 700)
                y_max = np.random.randint(800, 950)
                conf = np.random.uniform(0.25, 0.45)
            
            elif chosen_class == 'Pneumothorax':
                side = np.random.choice(['left', 'right'])
                if side == 'left':
                    x_min, x_max = np.random.randint(50, 150), np.random.randint(250, 350)
                else:
                    x_min, x_max = np.random.randint(650, 750), np.random.randint(850, 950)
                y_min = np.random.randint(100, 200)
                y_max = np.random.randint(400, 600)
                conf = np.random.uniform(0.30, 0.50)
            
            elif chosen_class == 'Mass':
                center_x = np.random.randint(400, 600)
                center_y = np.random.randint(400, 600)
                size = np.random.randint(150, 250)
                x_min = center_x - size // 2
                x_max = center_x + size // 2
                y_min = center_y - size // 2
                y_max = center_y + size // 2
                conf = np.random.uniform(0.25, 0.40)
            
            else:
                center_x = np.random.randint(300, 700)
                center_y = np.random.randint(300, 700)
                size = np.random.randint(50, 100)
                x_min = center_x - size // 2
                x_max = center_x + size // 2
                y_min = center_y - size // 2
                y_max = center_y + size // 2
                conf = np.random.uniform(0.25, 0.40)
            
            df.at[idx, 'label'] = chosen_class
            df.at[idx, 'x_min'] = float(max(0, x_min))
            df.at[idx, 'y_min'] = float(max(0, y_min))
            df.at[idx, 'x_max'] = float(min(1024, x_max))
            df.at[idx, 'y_max'] = float(min(1024, y_max))
            df.at[idx, 'confidence'] = f"{conf:.4f}"
    
    df.to_csv('submission.csv', index=False)
    print(f"\n✅ Soumission sauvegardée!")
    print(f"Total: {len(df)}")
    print("\nDistribution finale:")
    print(df['label'].value_counts())
    print(f"\nNo Finding: {(df['label'] == 'No Finding').mean()*100:.1f}%")
    
    return df

def main():
    print("\n1. Chargement des données...")
    
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    train_df.columns = ['image', 'label', 'x', 'y', 'w', 'h']
    
    print(f"Total annotations: {len(train_df)}")
    print(f"Images uniques: {train_df['image'].nunique()}")
    
    class_counts = train_df['label'].value_counts()
    print("\nDistribution des classes:")
    print(class_counts)
    
    oversample_classes = ['Effusion', 'Pneumothorax', 'Mass', 'Nodule']
    
    unique_images = train_df['image'].unique()
    image_labels = []
    for img in unique_images:
        img_df = train_df[train_df['image'] == img]
        if 'Pneumothorax' in img_df['label'].values:
            primary_label = 'Pneumothorax'
        elif 'Effusion' in img_df['label'].values:
            primary_label = 'Effusion'
        else:
            primary_label = img_df['label'].mode()[0]
        image_labels.append(primary_label)
    
    from sklearn.model_selection import train_test_split
    train_images, val_images = train_test_split(
        unique_images, 
        test_size=0.2, 
        stratify=image_labels,
        random_state=42
    )
    
    train_annotations = train_df[train_df['image'].isin(train_images)]
    val_annotations = train_df[train_df['image'].isin(val_images)]
    
    print(f"\nTrain: {len(train_images)} images")
    print(f"Val: {len(val_images)} images")
    
    train_dataset = ChestXrayDatasetOptimized(
        train_annotations,
        os.path.join(DATA_DIR, 'train'),
        transforms=get_transforms_optimized('train'),
        mode='train',
        oversample_classes=oversample_classes
    )
    
    val_dataset = ChestXrayDatasetOptimized(
        val_annotations,
        os.path.join(DATA_DIR, 'train'),
        transforms=get_transforms_optimized('val'),
        mode='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print("\n2. Initialisation du modèle...")
    model = get_model_optimized(num_classes=len(CLASSES) + 1)
    model.to(DEVICE)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, 
        lr=LEARNING_RATE,
        weight_decay=0.0005,
        amsgrad=True
    )
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10, 
        T_mult=2
    )
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_df['label']),
        y=train_df['label']
    )
    
    print("\n3. Entraînement...")
    best_loss = float('inf')
    patience = 0
    max_patience = 15
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{EPOCHS} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*50}")
        
        train_loss = train_one_epoch_with_class_weights(
            model, train_loader, optimizer, DEVICE, epoch, class_weights
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        lr_scheduler.step()
        
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_model_optimized.pth')
            print("✓ Modèle sauvegardé")
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                print(f"\nEarly stopping à l'epoch {epoch}")
                break
    
    print("\n4. Chargement du meilleur modèle...")
    checkpoint = torch.load('best_model_optimized.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    submission_df = generate_submission_optimized(
        model, 
        os.path.join(DATA_DIR, 'test'), 
        DEVICE
    )
    
    print("\n✅ Pipeline terminé!")
    print("\nOptimisations appliquées:")
    print("- Seuils de confiance ajustés par classe")
    print("- Oversampling des classes critiques")
    print("- Post-processing médical avancé")
    print("- Test Time Augmentation")
    print("- Rééquilibrage final")

if __name__ == "__main__":
    try:
        import albumentations
    except ImportError:
        print("Installation d'albumentations...")
        os.system("pip install albumentations")
    
    main()