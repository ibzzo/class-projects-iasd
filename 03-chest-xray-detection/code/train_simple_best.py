#!/usr/bin/env python3
"""
Simple but effective YOLOv8 training for chest X-ray detection
Optimized for best results with minimal complexity
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml
import torch

try:
    from ultralytics import YOLO
except ImportError:
    os.system("pip install ultralytics")
    from ultralytics import YOLO

# Configuration
DATA_DIR = Path("data")
DATASET_DIR = Path("yolo_dataset_simple")
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 30  # Quick training
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

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

print("=== Simple YOLOv8 Training ===")
print(f"Device: {DEVICE}")

def prepare_dataset():
    """Prepare dataset in YOLO format"""
    print("\n1. Preparing dataset...")
    
    # Create directories
    for split in ['train', 'val']:
        (DATASET_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df.columns = ['image', 'label', 'x', 'y', 'w', 'h']
    
    print(f"Total annotations: {len(train_df)}")
    print(f"Unique images: {train_df['image'].nunique()}")
    
    # Split data
    unique_images = train_df['image'].unique()
    train_images, val_images = train_test_split(unique_images, test_size=0.2, random_state=42)
    
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")
    
    # Process images
    for split, image_list in [('train', train_images), ('val', val_images)]:
        print(f"\nProcessing {split} set...")
        
        for img_name in tqdm(image_list):
            # Copy image
            src = DATA_DIR / 'train' / img_name
            dst = DATASET_DIR / 'images' / split / img_name
            
            if src.exists():
                shutil.copy2(src, dst)
                
                # Create label file
                img_annotations = train_df[train_df['image'] == img_name]
                label_path = DATASET_DIR / 'labels' / split / f"{Path(img_name).stem}.txt"
                
                with open(label_path, 'w') as f:
                    for _, ann in img_annotations.iterrows():
                        # Get class index
                        class_idx = CLASSES.index(ann['label']) if ann['label'] in CLASSES else 0
                        
                        # YOLO format: class x_center y_center width height (normalized)
                        # Assuming 1024x1024 images
                        img_size = 1024
                        x_center = (ann['x'] + ann['w'] / 2) / img_size
                        y_center = (ann['y'] + ann['h'] / 2) / img_size
                        width = ann['w'] / img_size
                        height = ann['h'] / img_size
                        
                        # Ensure valid values
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))
                        
                        f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def create_yaml():
    """Create YAML config file"""
    print("\n2. Creating YAML config...")
    
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
    """Train YOLOv8 model"""
    print("\n3. Training model...")
    
    # Use medium model for balance of speed and accuracy
    model = YOLO('yolov8m.pt')
    
    # Train with optimized parameters
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project='chest_xray_simple',
        name='best_model',
        patience=10,
        save=True,
        plots=True,
        # Key parameters
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        # Augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,  # No rotation for chest X-rays
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,  # No vertical flip
        fliplr=0.5,  # Horizontal flip OK
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        # Other
        conf=0.001,
        iou=0.5,
        max_det=10,
        amp=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        fraction=1.0,
        mask_ratio=4,
        dropout=0.0,
        val=True
    )
    
    print("\n✅ Training completed!")
    return model

def generate_submission(model):
    """Generate submission file"""
    print("\n4. Generating submission...")
    
    # Load test mapping
    id_mapping = pd.read_csv(DATA_DIR / 'ID_to_Image_Mapping.csv')
    test_dir = DATA_DIR / 'test'
    
    submission_data = []
    
    for idx, row in tqdm(id_mapping.iterrows(), total=len(id_mapping)):
        img_id = idx + 1
        img_name = row['image_id']
        img_path = test_dir / img_name
        
        if img_path.exists():
            # Run inference
            results = model(img_path, imgsz=IMG_SIZE, conf=0.1, iou=0.4, augment=True)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get best detection
                boxes = results[0].boxes
                best_idx = boxes.conf.argmax()
                
                # Extract prediction
                x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
                conf = float(boxes.conf[best_idx])
                cls = int(boxes.cls[best_idx])
                
                # Apply class-specific confidence threshold
                conf_thresholds = {
                    0: 0.15,  # Atelectasis
                    1: 0.20,  # Cardiomegaly
                    2: 0.10,  # Effusion
                    3: 0.15,  # Infiltrate
                    4: 0.15,  # Mass
                    5: 0.15,  # Nodule
                    6: 0.15,  # Pneumonia
                    7: 0.10   # Pneumothorax
                }
                
                if conf >= conf_thresholds.get(cls, 0.15):
                    label = CLASSES[cls] if cls < len(CLASSES) else 'Cardiomegaly'
                    
                    submission_data.append({
                        'id': img_id,
                        'image_id': img_name,
                        'x_min': round(float(x1), 2),
                        'y_min': round(float(y1), 2),
                        'x_max': round(float(x2), 2),
                        'y_max': round(float(y2), 2),
                        'confidence': f"{conf:.4f}",
                        'label': label
                    })
                else:
                    # Low confidence - No Finding
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
                # No detection
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
            # Image not found
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
    df = pd.DataFrame(submission_data)
    df = df[['id', 'image_id', 'x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'label']]
    
    # Apply post-processing to balance classes
    print("\n5. Post-processing predictions...")
    
    # Count current distribution
    class_counts = df[df['label'] != 'No Finding']['label'].value_counts()
    
    # If too many Cardiomegaly, reduce some low-confidence ones
    if 'Cardiomegaly' in class_counts and class_counts['Cardiomegaly'] > 50:
        cardio_mask = (df['label'] == 'Cardiomegaly') & (df['confidence'].astype(float) < 0.3)
        df.loc[cardio_mask, 'label'] = 'No Finding'
        df.loc[cardio_mask, 'confidence'] = "0.0000"
    
    # Save submission
    df.to_csv('submission.csv', index=False)
    
    print(f"\n✅ Submission saved to submission.csv")
    print(f"Total predictions: {len(df)}")
    print(f"No Finding: {(df['label'] == 'No Finding').sum()}")
    print("\nClass distribution:")
    print(df['label'].value_counts())

def main():
    """Main pipeline"""
    # Prepare dataset
    prepare_dataset()
    
    # Create config
    yaml_path = create_yaml()
    
    # Train model
    model = train_model(yaml_path)
    
    # Generate submission
    generate_submission(model)
    
    print("\n=== Training complete! ===")
    print("Submission.csv is ready for Kaggle!")

if __name__ == "__main__":
    main()