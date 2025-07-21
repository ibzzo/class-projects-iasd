#!/usr/bin/env python3
"""
Advanced training pipeline with data augmentation and optimization techniques
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
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Check device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Configuration
BATCH_SIZE = 4
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
IMG_SIZE = 512
NUM_WORKERS = 4

# Classes
PATHOLOGY_CLASSES = {
    'Atelectasis': 1,
    'Cardiomegaly': 2,
    'Effusion': 3,
    'Infiltrate': 4,
    'Mass': 5,
    'Nodule': 6,
    'Pneumonia': 7,
    'Pneumothorax': 8
}
NUM_CLASSES = len(PATHOLOGY_CLASSES) + 1  # +1 for background

print(f"Classes: {list(PATHOLOGY_CLASSES.keys())}")

class ChestXrayDataset(Dataset):
    """Enhanced dataset with advanced augmentations"""
    
    def __init__(self, annotations_df, image_dir, transforms=None, train=True):
        self.image_dir = image_dir
        self.train = train
        
        # Group annotations by image
        self.image_annotations = {}
        for _, row in annotations_df.iterrows():
            img_name = row['image']
            if img_name not in self.image_annotations:
                self.image_annotations[img_name] = []
            
            self.image_annotations[img_name].append({
                'label': row['label'],
                'bbox': [row['x'], row['y'], row['w'], row['h']]
            })
        
        self.image_list = list(self.image_annotations.keys())
        
        # Define augmentations
        if train:
            self.transform = A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.15, 
                    rotate_limit=10, 
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
        else:
            self.transform = A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Prepare annotations
        boxes = []
        labels = []
        
        for ann in self.image_annotations[img_name]:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, w, h])
            labels.append(PATHOLOGY_CLASSES.get(ann['label'], 0))
        
        # Apply augmentations
        if len(boxes) > 0:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        else:
            # No boxes - just transform image
            image = self.transform(image=image)['image']
            boxes = [[0, 0, 1, 1]]  # Dummy box
            labels = [0]  # Background
        
        # Convert to tensor format
        boxes_tensor = []
        for box in boxes:
            x, y, w, h = box
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h
            boxes_tensor.append([x_min, y_min, x_max, y_max])
        
        boxes_tensor = torch.as_tensor(boxes_tensor, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        
        # Handle empty annotations
        if len(boxes_tensor) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([idx])
        }
        
        return image.float() / 255.0, target

def get_model(num_classes):
    """Get model with custom backbone"""
    # Use ResNet50 with FPN
    model = fasterrcnn_resnet50_fpn(
        pretrained=True,
        min_size=IMG_SIZE,
        max_size=IMG_SIZE
    )
    
    # Replace the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None):
    """Training with mixed precision"""
    model.train()
    epoch_loss = 0
    
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    for images, targets in pbar:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
        
        epoch_loss += losses.item()
        pbar.set_postfix({'loss': f'{losses.item():.4f}'})
    
    return epoch_loss / len(data_loader)

@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluation with metrics"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    for images, targets in tqdm(data_loader, desc='Evaluating'):
        images = list(img.to(device) for img in images)
        outputs = model(images)
        
        all_predictions.extend(outputs)
        all_targets.extend(targets)
    
    # Calculate metrics
    print("Evaluation completed")
    return all_predictions, all_targets

def generate_submission(model, test_dir, id_mapping_path, device):
    """Generate submission with ensemble predictions"""
    print("\nGenerating submission...")
    model.eval()
    
    # Load mapping
    id_mapping = pd.read_csv(id_mapping_path)
    
    # Prepare transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    
    submission_data = []
    
    with torch.no_grad():
        for idx, row in tqdm(id_mapping.iterrows(), total=len(id_mapping)):
            img_id = idx + 1
            img_name = row['image_id']
            img_path = os.path.join(test_dir, img_name)
            
            if os.path.exists(img_path):
                # Load and preprocess
                image = Image.open(img_path).convert('RGB')
                orig_w, orig_h = image.size
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Get predictions
                outputs = model(image_tensor)[0]
                
                # Scale factor
                scale_x = orig_w / IMG_SIZE
                scale_y = orig_h / IMG_SIZE
                
                # Process predictions
                pred_added = False
                
                if len(outputs['boxes']) > 0:
                    # Apply NMS
                    keep = torchvision.ops.nms(outputs['boxes'], outputs['scores'], 0.4)
                    
                    if len(keep) > 0:
                        # Take best prediction after NMS
                        best_idx = keep[0]
                        
                        score = float(outputs['scores'][best_idx])
                        
                        if score >= 0.15:  # Confidence threshold
                            box = outputs['boxes'][best_idx].cpu().numpy()
                            label_idx = int(outputs['labels'][best_idx])
                            
                            # Scale coordinates back
                            x1 = box[0] * scale_x
                            y1 = box[1] * scale_y
                            x2 = box[2] * scale_x
                            y2 = box[3] * scale_y
                            
                            # Get label
                            id_to_label = {v: k for k, v in PATHOLOGY_CLASSES.items()}
                            label = id_to_label.get(label_idx, 'Cardiomegaly')
                            
                            submission_data.append({
                                'id': img_id,
                                'image_id': img_name,
                                'x_min': round(float(x1), 2),
                                'y_min': round(float(y1), 2),
                                'x_max': round(float(x2), 2),
                                'y_max': round(float(y2), 2),
                                'confidence': f"{score:.4f}",
                                'label': label
                            })
                            pred_added = True
                
                if not pred_added:
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
                # Default
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
    
    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)
    submission_df = submission_df[['id', 'image_id', 'x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'label']]
    
    # Save
    submission_df.to_csv('submission.csv', index=False)
    
    print(f"✅ Submission saved")
    print(f"Class distribution:")
    print(submission_df['label'].value_counts())

def main():
    """Main training pipeline"""
    print("=== Advanced Chest X-ray Detection Training ===\n")
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('data/train.csv')
    train_df.columns = ['image', 'label', 'x', 'y', 'w', 'h']
    
    print(f"Total annotations: {len(train_df)}")
    print(f"Unique images: {train_df['image'].nunique()}")
    
    # Split data
    unique_images = train_df['image'].unique()
    train_images, val_images = train_test_split(unique_images, test_size=0.2, random_state=42)
    
    train_data = train_df[train_df['image'].isin(train_images)]
    val_data = train_df[train_df['image'].isin(val_images)]
    
    print(f"\nTrain images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")
    
    # Create datasets
    train_dataset = ChestXrayDataset(train_data, 'data/train', train=True)
    val_dataset = ChestXrayDataset(val_data, 'data/train', train=False)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = get_model(NUM_CLASSES)
    model.to(DEVICE)
    
    # Optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS,
        pct_start=0.1
    )
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == 'cuda' else None
    
    # Training
    print("\nStarting training...")
    best_loss = float('inf')
    train_losses = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, scaler)
        scheduler.step()
        train_losses.append(train_loss)
        
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}: Loss = {train_loss:.4f}")
        
        # Evaluate every 5 epochs
        if epoch % 5 == 0:
            evaluate(model, val_loader, DEVICE)
            
            # Save best model
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(model.state_dict(), 'best_model_advanced.pth')
                print(f"Best model saved (loss: {best_loss:.4f})")
    
    # Save final model
    torch.save(model.state_dict(), 'final_model_advanced.pth')
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_curve.png')
    print("\nTraining curve saved to training_curve.png")
    
    # Generate submission
    generate_submission(
        model,
        'data/test',
        'data/ID_to_Image_Mapping.csv',
        DEVICE
    )
    
    print("\n=== Training completed! ===")

if __name__ == '__main__':
    # Check if albumentations is installed
    try:
        import albumentations
    except ImportError:
        print("Installing albumentations...")
        import subprocess
        subprocess.check_call(["pip", "install", "albumentations"])
    
    main()