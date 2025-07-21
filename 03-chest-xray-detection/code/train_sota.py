#!/usr/bin/env python3
"""
State-of-the-Art Chest X-ray Detection Pipeline
Combines multiple strategies for best performance
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import yaml
import json
import torch
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# Check installations
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    subprocess.check_call(["pip", "install", "ultralytics>=8.0.0"])
    from ultralytics import YOLO

# Configuration
PROJECT_DIR = Path(".")
DATA_DIR = PROJECT_DIR / "data"
DATASET_DIR = PROJECT_DIR / "yolo_dataset_sota"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

# Advanced training parameters
IMG_SIZES = [640, 800, 1024]  # Multi-scale training
BATCH_SIZE = 16  # Increase if possible
EPOCHS = 100
PATIENCE = 20
DEVICE = 0 if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Classes with weights based on frequency
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

# Advanced augmentation parameters
AUGMENTATION_CONFIG = {
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 5,  # Small rotation for medical images
    'translate': 0.1,
    'scale': 0.2,
    'shear': 2,
    'perspective': 0.0001,
    'flipud': 0.0,  # No vertical flip for chest x-rays
    'fliplr': 0.5,
    'mosaic': 0.5,  # Reduced mosaic for medical images
    'mixup': 0.2,
    'copy_paste': 0.1,
    'auto_augment': 'randaugment'
}

print("=== State-of-the-Art Chest X-ray Detection ===\n")
print(f"Device: {DEVICE}")
print(f"Multi-scale training: {IMG_SIZES}")

def analyze_dataset():
    """Analyze dataset for insights"""
    print("\n1. Analyzing dataset...")
    
    # Load annotations
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df.columns = ['image', 'label', 'x', 'y', 'w', 'h']
    
    # Statistics
    print(f"Total annotations: {len(train_df)}")
    print(f"Unique images: {train_df['image'].nunique()}")
    
    # Class distribution
    class_counts = train_df['label'].value_counts()
    print("\nClass distribution:")
    print(class_counts)
    
    # Calculate class weights for balanced training
    total_samples = len(train_df)
    class_weights = {}
    for cls in CLASSES:
        count = class_counts.get(cls, 1)
        weight = np.sqrt(total_samples / (len(CLASSES) * count))
        class_weights[cls] = round(weight, 3)
    
    print("\nClass weights for balanced training:")
    for cls, weight in class_weights.items():
        print(f"  {cls}: {weight}")
    
    # Analyze box sizes
    train_df['area'] = train_df['w'] * train_df['h']
    train_df['aspect_ratio'] = train_df['w'] / train_df['h'].clip(lower=1)
    
    print("\nBounding box statistics:")
    print(f"  Average area: {train_df['area'].mean():.0f} pixels²")
    print(f"  Average aspect ratio: {train_df['aspect_ratio'].mean():.2f}")
    
    # Identify small, medium, large objects
    small_threshold = train_df['area'].quantile(0.33)
    large_threshold = train_df['area'].quantile(0.67)
    
    print(f"\nObject size distribution:")
    print(f"  Small (<{small_threshold:.0f}): {(train_df['area'] < small_threshold).sum()}")
    print(f"  Medium: {((train_df['area'] >= small_threshold) & (train_df['area'] < large_threshold)).sum()}")
    print(f"  Large (>{large_threshold:.0f}): {(train_df['area'] >= large_threshold).sum()}")
    
    return train_df, class_weights

def create_stratified_splits(train_df, n_folds=5):
    """Create stratified K-fold splits"""
    print(f"\n2. Creating {n_folds}-fold stratified splits...")
    
    # Group by image and get primary label
    image_labels = train_df.groupby('image')['label'].agg(lambda x: x.mode()[0])
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    folds = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(image_labels.index, image_labels.values)):
        train_images = image_labels.index[train_idx].tolist()
        val_images = image_labels.index[val_idx].tolist()
        folds.append({
            'fold': fold,
            'train': train_images,
            'val': val_images
        })
        print(f"  Fold {fold}: Train {len(train_images)}, Val {len(val_images)}")
    
    return folds

def prepare_yolo_dataset_advanced(train_df, fold_data):
    """Prepare dataset with advanced preprocessing"""
    print(f"\n3. Preparing YOLO dataset for fold {fold_data['fold']}...")
    
    # Create directories
    fold_dir = DATASET_DIR / f"fold_{fold_data['fold']}"
    for split in ['train', 'val']:
        (fold_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (fold_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Process images
    for split, image_list in [('train', fold_data['train']), ('val', fold_data['val'])]:
        print(f"  Processing {split} set...")
        
        for img_name in tqdm(image_list, desc=f"Processing {split}"):
            # Copy image
            src_img = DATA_DIR / 'train' / img_name
            dst_img = fold_dir / 'images' / split / img_name
            
            if src_img.exists():
                # Apply CLAHE preprocessing for better contrast
                img = cv2.imread(str(src_img), cv2.IMREAD_GRAYSCALE)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img_enhanced = clahe.apply(img)
                img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2RGB)
                cv2.imwrite(str(dst_img), img_enhanced)
                
                # Create YOLO format label file
                img_annotations = train_df[train_df['image'] == img_name]
                label_path = fold_dir / 'labels' / split / f"{Path(img_name).stem}.txt"
                
                with open(label_path, 'w') as f:
                    for _, ann in img_annotations.iterrows():
                        # Get class index
                        class_idx = CLASS_DICT.get(ann['label'], 0)
                        
                        # Convert to YOLO format
                        img_h, img_w = img.shape[:2]
                        
                        # Convert from [x, y, w, h] to [x_center, y_center, w, h] normalized
                        x_center = (ann['x'] + ann['w'] / 2) / img_w
                        y_center = (ann['y'] + ann['h'] / 2) / img_h
                        width = ann['w'] / img_w
                        height = ann['h'] / img_h
                        
                        # Ensure valid values
                        x_center = np.clip(x_center, 0, 1)
                        y_center = np.clip(y_center, 0, 1)
                        width = np.clip(width, 0, 1)
                        height = np.clip(height, 0, 1)
                        
                        f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    return fold_dir

def create_yaml_config_advanced(fold_dir, fold_num, class_weights):
    """Create advanced YAML configuration"""
    print(f"\n4. Creating YAML configuration for fold {fold_num}...")
    
    config = {
        'path': str(fold_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(CLASSES),
        'names': CLASSES,
        # Advanced settings
        'class_weights': [class_weights.get(cls, 1.0) for cls in CLASSES]
    }
    
    yaml_path = fold_dir / f'chest_xray_fold{fold_num}.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    return yaml_path

def train_model_advanced(yaml_path, fold_num, img_size=640):
    """Train YOLOv8x model with advanced techniques"""
    print(f"\n5. Training YOLOv8x for fold {fold_num}, size {img_size}...")
    
    # Use larger model for better performance
    model = YOLO('yolov8x.pt')  # Extra large model
    
    # Advanced training parameters
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=img_size,
        batch=BATCH_SIZE,
        device=DEVICE,
        project='chest_xray_sota',
        name=f'fold{fold_num}_size{img_size}',
        patience=PATIENCE,
        save=True,
        plots=True,
        # Detection settings
        conf=0.001,
        iou=0.5,
        max_det=20,  # More detections per image
        # Optimization
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # Augmentations
        **AUGMENTATION_CONFIG,
        # Advanced settings
        label_smoothing=0.1,
        nms_time_limit=30.0,
        save_period=10,
        close_mosaic=int(EPOCHS * 0.9),  # Disable mosaic for final epochs
        amp=True,  # Mixed precision
        fraction=1.0,  # Use full dataset
        multi_scale=True,  # Multi-scale training
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.1,
    )
    
    return f'chest_xray_sota/fold{fold_num}_size{img_size}/weights/best.pt'

def ensemble_predictions(model_paths, test_dir, id_mapping_path):
    """Generate ensemble predictions from multiple models"""
    print("\n6. Generating ensemble predictions...")
    
    # Load models
    models = []
    for path in model_paths:
        if os.path.exists(path):
            models.append(YOLO(path))
            print(f"  Loaded: {path}")
    
    if not models:
        print("ERROR: No models found!")
        return
    
    # Load test mapping
    id_mapping = pd.read_csv(id_mapping_path)
    
    # Prepare submission data
    submission_data = []
    
    # Process each test image
    for idx, row in tqdm(id_mapping.iterrows(), total=len(id_mapping), desc="Ensemble inference"):
        img_id = idx + 1
        img_name = row['image_id']
        img_path = os.path.join(test_dir, img_name)
        
        if os.path.exists(img_path):
            # Collect predictions from all models
            all_boxes = []
            all_scores = []
            all_classes = []
            
            # Apply CLAHE preprocessing
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_enhanced = clahe.apply(img)
            img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2RGB)
            
            # Save enhanced image temporarily
            temp_path = f"/tmp/{img_name}"
            cv2.imwrite(temp_path, img_enhanced)
            
            for model in models:
                # Run inference at multiple scales
                for scale in [0.8, 1.0, 1.2]:
                    size = int(640 * scale)
                    results = model(temp_path, imgsz=size, conf=0.1, iou=0.4, max_det=10, augment=True)
                    
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        boxes = results[0].boxes
                        all_boxes.extend(boxes.xyxy.cpu().numpy())
                        all_scores.extend(boxes.conf.cpu().numpy())
                        all_classes.extend(boxes.cls.cpu().numpy().astype(int))
            
            # Remove temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Apply Weighted Boxes Fusion or NMS
            if len(all_boxes) > 0:
                # Convert to numpy arrays
                all_boxes = np.array(all_boxes)
                all_scores = np.array(all_scores)
                all_classes = np.array(all_classes)
                
                # Group by class and apply NMS
                best_box = None
                best_score = 0
                best_class = 0
                
                for cls in range(len(CLASSES)):
                    cls_mask = all_classes == cls
                    if cls_mask.any():
                        cls_boxes = all_boxes[cls_mask]
                        cls_scores = all_scores[cls_mask]
                        
                        # Get best box for this class
                        if len(cls_scores) > 0:
                            best_idx = cls_scores.argmax()
                            if cls_scores[best_idx] > best_score:
                                best_score = cls_scores[best_idx]
                                best_box = cls_boxes[best_idx]
                                best_class = cls
                
                # Add best prediction
                if best_box is not None and best_score > 0.15:
                    x1, y1, x2, y2 = best_box
                    label = CLASSES[best_class]
                    
                    submission_data.append({
                        'id': img_id,
                        'image_id': img_name,
                        'x_min': round(float(x1), 2),
                        'y_min': round(float(y1), 2),
                        'x_max': round(float(x2), 2),
                        'y_max': round(float(y2), 2),
                        'confidence': f"{best_score:.4f}",
                        'label': label
                    })
                else:
                    # No confident detection
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
                # No detection at all
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
    submission_df = pd.DataFrame(submission_data)
    submission_df = submission_df[['id', 'image_id', 'x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'label']]
    
    # Save submission
    submission_df.to_csv('submission_sota.csv', index=False)
    
    print(f"\n✅ Submission saved to submission_sota.csv")
    print(f"Total predictions: {len(submission_df)}")
    print(f"\nClass distribution:")
    print(submission_df['label'].value_counts())
    
    # Also save with original name
    submission_df.to_csv('submission.csv', index=False)
    print("Also saved as submission.csv")

def post_process_submission():
    """Apply post-processing heuristics"""
    print("\n7. Applying post-processing...")
    
    # Load submission
    df = pd.read_csv('submission.csv')
    
    # Heuristics based on medical knowledge
    for idx, row in df.iterrows():
        # Adjust confidence based on box size
        area = (row['x_max'] - row['x_min']) * (row['y_max'] - row['y_min'])
        
        # Very small boxes are likely false positives
        if area < 10000 and row['label'] != 'Nodule':
            df.at[idx, 'confidence'] = str(float(row['confidence']) * 0.5)
        
        # Very large boxes for certain classes
        if area > 400000 and row['label'] in ['Nodule', 'Mass']:
            df.at[idx, 'confidence'] = str(float(row['confidence']) * 0.7)
    
    # Save processed submission
    df.to_csv('submission.csv', index=False)
    print("Post-processing completed!")

def main():
    """Main training pipeline"""
    
    # Step 1: Analyze dataset
    train_df, class_weights = analyze_dataset()
    
    # Step 2: Create folds
    folds = create_stratified_splits(train_df, n_folds=3)  # 3 folds for faster training
    
    # Step 3: Train models
    model_paths = []
    
    for fold_data in folds[:2]:  # Train first 2 folds
        # Prepare dataset
        fold_dir = prepare_yolo_dataset_advanced(train_df, fold_data)
        
        # Create config
        yaml_path = create_yaml_config_advanced(fold_dir, fold_data['fold'], class_weights)
        
        # Train at multiple scales
        for img_size in [640, 800]:  # Two scales
            model_path = train_model_advanced(yaml_path, fold_data['fold'], img_size)
            model_paths.append(model_path)
    
    # Step 4: Generate ensemble predictions
    ensemble_predictions(model_paths, DATA_DIR / 'test', DATA_DIR / 'ID_to_Image_Mapping.csv')
    
    # Step 5: Post-process
    post_process_submission()
    
    print("\n=== SOTA Pipeline completed! ===")
    print("\nKey techniques used:")
    print("1. YOLOv8x (extra large model)")
    print("2. Multi-scale training and inference")
    print("3. Stratified K-fold cross-validation")
    print("4. CLAHE preprocessing")
    print("5. Class-balanced training")
    print("6. Model ensemble")
    print("7. Test-time augmentation")
    print("8. Post-processing heuristics")
    print("\nsubmission.csv is ready for Kaggle!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick submission with existing models
        model_paths = []
        for fold in range(3):
            for size in [640, 800]:
                path = f'chest_xray_sota/fold{fold}_size{size}/weights/best.pt'
                if os.path.exists(path):
                    model_paths.append(path)
        
        if model_paths:
            ensemble_predictions(model_paths, Path('data/test'), Path('data/ID_to_Image_Mapping.csv'))
            post_process_submission()
        else:
            print("No trained models found! Run full training first.")
    else:
        # Install required packages
        os.system("pip install opencv-python scikit-learn matplotlib seaborn -q")
        # Full training pipeline
        main()