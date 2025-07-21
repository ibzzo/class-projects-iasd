#!/usr/bin/env python3
"""
Complete YOLOv8 training pipeline for Chest X-ray Detection
Based on the exemple.ipynb approach
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

# Check if ultralytics is installed
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    subprocess.check_call(["pip", "install", "ultralytics"])
    from ultralytics import YOLO

# Configuration
PROJECT_DIR = Path(".")
DATA_DIR = PROJECT_DIR / "data"
DATASET_DIR = PROJECT_DIR / "yolo_dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

# Training parameters
IMG_SIZE = 640  # Standard YOLO size, can increase to 1024 if GPU allows
BATCH_SIZE = 8
EPOCHS = 50
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
CLASS_DICT = {cls: idx for idx, cls in enumerate(CLASSES)}

print("=== YOLOv8 Chest X-ray Detection Training ===\n")
print(f"Device: {DEVICE}")
print(f"Classes: {CLASSES}")

def prepare_yolo_dataset():
    """Prepare dataset in YOLO format"""
    print("\n1. Preparing YOLO dataset structure...")
    
    # Create directories
    for split in ['train', 'val']:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df.columns = ['image', 'label', 'x', 'y', 'w', 'h']
    
    print(f"Total annotations: {len(train_df)}")
    print(f"Unique images: {train_df['image'].nunique()}")
    print("\nLabel distribution:")
    print(train_df['label'].value_counts())
    
    # Split by images (not annotations)
    unique_images = train_df['image'].unique()
    train_images, val_images = train_test_split(
        unique_images, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"\nTrain images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")
    
    # Process images
    for split, image_list in [('train', train_images), ('val', val_images)]:
        print(f"\nProcessing {split} set...")
        
        for img_name in tqdm(image_list, desc=f"Copying {split} images"):
            # Copy image
            src_img = DATA_DIR / 'train' / img_name
            dst_img = IMAGES_DIR / split / img_name
            
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
                
                # Create YOLO format label file
                img_annotations = train_df[train_df['image'] == img_name]
                label_path = LABELS_DIR / split / f"{Path(img_name).stem}.txt"
                
                with open(label_path, 'w') as f:
                    for _, ann in img_annotations.iterrows():
                        # Get class index
                        class_idx = CLASS_DICT.get(ann['label'], 0)
                        
                        # Convert to YOLO format (normalized)
                        # Assuming images are 1024x1024
                        img_w = img_h = 1024
                        
                        # Convert from [x, y, w, h] to [x_center, y_center, w, h] normalized
                        x_center = (ann['x'] + ann['w'] / 2) / img_w
                        y_center = (ann['y'] + ann['h'] / 2) / img_h
                        width = ann['w'] / img_w
                        height = ann['h'] / img_h
                        
                        # Ensure values are in [0, 1]
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))
                        
                        f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def create_yaml_config():
    """Create YAML configuration file for YOLOv8"""
    print("\n2. Creating YAML configuration...")
    
    config = {
        'path': str(DATASET_DIR.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(CLASSES),
        'names': CLASSES
    }
    
    yaml_path = DATASET_DIR / 'chest_xray.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    print(f"YAML config saved to: {yaml_path}")
    return yaml_path

def train_model(yaml_path):
    """Train YOLOv8 model"""
    print("\n3. Training YOLOv8 model...")
    
    # Initialize model
    model = YOLO('yolov8s.pt')  # Start with small model
    
    # Train
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project='chest_xray_detection',
        name='yolov8s_run',
        patience=10,
        save=True,
        plots=True,
        conf=0.001,  # Low confidence for validation to see all predictions
        iou=0.5,
        max_det=10,  # Max detections per image
        amp=False,  # Disable mixed precision if issues
        workers=4,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0,
        translate=0.1,
        scale=0.5,
        shear=0,
        perspective=0,
        flipud=0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0
    )
    
    print("\n✅ Training completed!")
    print(f"Best model saved to: chest_xray_detection/yolov8s_run/weights/best.pt")
    
    return model

def validate_model(model, yaml_path):
    """Validate the trained model"""
    print("\n4. Validating model...")
    
    # Run validation
    metrics = model.val(
        data=str(yaml_path),
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        conf=0.001,
        iou=0.5,
        device=DEVICE
    )
    
    print("\nValidation Results:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    # Class-wise AP
    print("\nClass-wise Average Precision:")
    for i, class_name in enumerate(CLASSES):
        if i < len(metrics.box.ap50):
            print(f"{class_name}: {metrics.box.ap50[i]:.4f}")

def generate_submission(model_path='chest_xray_detection/yolov8s_run/weights/best.pt'):
    """Generate submission file using trained model"""
    print("\n5. Generating submission file...")
    
    # Load model
    model = YOLO(model_path)
    
    # Load test image mapping
    id_mapping = pd.read_csv(DATA_DIR / 'ID_to_Image_Mapping.csv')
    
    # Prepare submission data
    submission_data = []
    
    # Process each test image
    for idx, row in tqdm(id_mapping.iterrows(), total=len(id_mapping), desc="Generating predictions"):
        img_id = idx + 1
        img_name = row['image_id']
        img_path = DATA_DIR / 'test' / img_name
        
        if img_path.exists():
            # Run inference
            results = model(img_path, imgsz=IMG_SIZE, conf=0.15, iou=0.5, max_det=5)
            
            # Get predictions
            pred_added = False
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                
                # Sort by confidence and take best prediction
                if boxes.conf.numel() > 0:
                    best_idx = boxes.conf.argmax()
                    
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
                    conf = float(boxes.conf[best_idx].cpu())
                    cls = int(boxes.cls[best_idx].cpu())
                    
                    # Get class name
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
                    pred_added = True
            
            # Add "No Finding" if no detection
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
            # Default if image not found
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
    
    # Save submission
    submission_df.to_csv('submission.csv', index=False)
    
    print(f"\n✅ Submission saved to submission.csv")
    print(f"Total predictions: {len(submission_df)}")
    print(f"\nClass distribution:")
    print(submission_df['label'].value_counts())

def main():
    """Main training pipeline"""
    
    # Step 1: Prepare dataset
    prepare_yolo_dataset()
    
    # Step 2: Create config
    yaml_path = create_yaml_config()
    
    # Step 3: Train model
    model = train_model(yaml_path)
    
    # Step 4: Validate model
    validate_model(model, yaml_path)
    
    # Step 5: Generate submission
    generate_submission()
    
    print("\n=== Pipeline completed successfully! ===")
    print("\nNext steps:")
    print("1. Check training plots in chest_xray_detection/yolov8s_run/")
    print("2. Review submission.csv")
    print("3. Submit to Kaggle")
    print("\nTo generate a new submission with the trained model:")
    print("python train_yolov8.py --inference-only")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--inference-only":
        # Just generate submission with existing model
        generate_submission()
    else:
        # Full training pipeline
        main()