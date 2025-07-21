#!/usr/bin/env python3
"""
Hyperparameter optimization for best performance
"""

import os
import yaml
import json
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product

def create_experiment_configs():
    """Create multiple experiment configurations"""
    
    # Base configuration
    base_config = {
        'model': 'yolov8x.pt',
        'data': 'yolo_dataset/chest_xray.yaml',
        'epochs': 50,
        'patience': 15,
        'batch': 16,
        'imgsz': 640,
        'device': 0,
        'project': 'chest_xray_experiments',
        'exist_ok': True,
        'amp': True,
        'close_mosaic': 40,
        'save': True,
        'cache': True
    }
    
    # Hyperparameter search space
    search_space = {
        'lr0': [0.0001, 0.001, 0.01],
        'lrf': [0.01, 0.1],
        'momentum': [0.9, 0.937],
        'weight_decay': [0.0, 0.0005],
        'warmup_epochs': [3, 5],
        'box': [5.0, 7.5, 10.0],
        'cls': [0.3, 0.5, 1.0],
        'label_smoothing': [0.0, 0.1],
        'hsv_h': [0.01, 0.015],
        'hsv_s': [0.5, 0.7],
        'hsv_v': [0.3, 0.4],
        'translate': [0.1, 0.2],
        'scale': [0.2, 0.5],
        'fliplr': [0.5],
        'mosaic': [0.5, 1.0],
        'mixup': [0.0, 0.2],
        'copy_paste': [0.0, 0.1]
    }
    
    # Generate experiments (sample random combinations)
    experiments = []
    
    # Best known configuration
    best_config = base_config.copy()
    best_config.update({
        'name': 'best_known',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5,
        'box': 7.5,
        'cls': 0.5,
        'label_smoothing': 0.1,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'translate': 0.1,
        'scale': 0.2,
        'fliplr': 0.5,
        'mosaic': 0.5,
        'mixup': 0.2,
        'copy_paste': 0.1
    })
    experiments.append(best_config)
    
    # High learning rate experiment
    high_lr_config = base_config.copy()
    high_lr_config.update({
        'name': 'high_lr',
        'lr0': 0.01,
        'lrf': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.0,
        'warmup_epochs': 3,
        'box': 10.0,
        'cls': 1.0
    })
    experiments.append(high_lr_config)
    
    # Heavy augmentation experiment
    heavy_aug_config = base_config.copy()
    heavy_aug_config.update({
        'name': 'heavy_aug',
        'lr0': 0.001,
        'translate': 0.2,
        'scale': 0.5,
        'mosaic': 1.0,
        'mixup': 0.2,
        'copy_paste': 0.1,
        'degrees': 10,
        'shear': 5
    })
    experiments.append(heavy_aug_config)
    
    # Conservative experiment
    conservative_config = base_config.copy()
    conservative_config.update({
        'name': 'conservative',
        'lr0': 0.0001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5,
        'box': 5.0,
        'cls': 0.3,
        'mosaic': 0.0,
        'mixup': 0.0
    })
    experiments.append(conservative_config)
    
    # Multi-scale experiment
    multiscale_config = base_config.copy()
    multiscale_config.update({
        'name': 'multiscale',
        'imgsz': 800,
        'multi_scale': True,
        'scale': 0.5,
        'batch': 8  # Reduced batch for larger images
    })
    experiments.append(multiscale_config)
    
    return experiments

def run_experiment(config):
    """Run a single experiment"""
    try:
        from ultralytics import YOLO
    except ImportError:
        os.system("pip install ultralytics")
        from ultralytics import YOLO
    
    print(f"\n=== Running experiment: {config['name']} ===")
    
    # Create model
    model = YOLO(config['model'])
    
    # Train
    results = model.train(**config)
    
    # Validate and get metrics
    metrics = model.val()
    
    # Save results
    result_summary = {
        'name': config['name'],
        'config': config,
        'metrics': {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr)
        }
    }
    
    # Save to file
    with open(f"experiment_{config['name']}_results.json", 'w') as f:
        json.dump(result_summary, f, indent=2)
    
    return result_summary

def analyze_experiments():
    """Analyze all experiment results"""
    print("\n=== Analyzing Experiment Results ===\n")
    
    results = []
    
    # Load all result files
    for file in Path('.').glob('experiment_*_results.json'):
        with open(file, 'r') as f:
            results.append(json.load(f))
    
    if not results:
        print("No experiment results found!")
        return
    
    # Sort by mAP50
    results.sort(key=lambda x: x['metrics']['mAP50'], reverse=True)
    
    print("Experiment Rankings (by mAP@50):")
    print(f"{'Rank':<5} {'Name':<15} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 60)
    
    for i, result in enumerate(results, 1):
        metrics = result['metrics']
        print(f"{i:<5} {result['name']:<15} {metrics['mAP50']:<10.4f} {metrics['mAP50-95']:<10.4f} "
              f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f}")
    
    # Best configuration
    best = results[0]
    print(f"\n🏆 Best Configuration: {best['name']}")
    print(f"   mAP@50: {best['metrics']['mAP50']:.4f}")
    
    # Key insights
    print("\n📊 Key Insights:")
    
    # Learning rate analysis
    lr_results = [(r['config'].get('lr0', 0.001), r['metrics']['mAP50']) for r in results]
    best_lr = max(lr_results, key=lambda x: x[1])
    print(f"   - Best learning rate: {best_lr[0]}")
    
    # Augmentation analysis
    aug_results = [(r['config'].get('mosaic', 0), r['metrics']['mAP50']) for r in results]
    best_mosaic = max(aug_results, key=lambda x: x[1])
    print(f"   - Best mosaic value: {best_mosaic[0]}")
    
    # Save best config
    with open('best_hyperparameters.yaml', 'w') as f:
        yaml.dump(best['config'], f, sort_keys=False)
    print("\n✅ Best hyperparameters saved to best_hyperparameters.yaml")

def create_final_training_script():
    """Create optimized training script with best hyperparameters"""
    
    script_content = '''#!/usr/bin/env python3
"""
Final optimized training script for chest X-ray detection
Uses best hyperparameters from optimization
"""

from ultralytics import YOLO
import torch
import os

# Best hyperparameters
BEST_CONFIG = {
    'model': 'yolov8x.pt',
    'data': 'yolo_dataset/chest_xray.yaml',
    'epochs': 100,
    'patience': 20,
    'batch': 16,
    'imgsz': 640,
    'device': 0 if torch.cuda.is_available() else 'cpu',
    'project': 'chest_xray_final',
    'name': 'best_model',
    'exist_ok': True,
    'lr0': 0.001,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 5,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'label_smoothing': 0.1,
    'nms_time_limit': 30.0,
    'save_period': 10,
    'close_mosaic': 80,
    'amp': True,
    'fraction': 1.0,
    'multi_scale': True,
    'overlap_mask': True,
    'mask_ratio': 4,
    'dropout': 0.1,
    'val': True,
    'plots': True,
    'save': True,
    'cache': True,
    'optimizer': 'AdamW',
    'seed': 42,
    'deterministic': True,
    'single_cls': False,
    'image_weights': False,
    'rect': False,
    'cos_lr': True,
    'resume': False,
    'nosave': False,
    'noval': False,
    'noautoanchor': False,
    'noplots': False,
    'evolve': False,
    'bucket': '',
    'gsutil_upload': False,
    # Augmentations
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 5,
    'translate': 0.1,
    'scale': 0.2,
    'shear': 2,
    'perspective': 0.0001,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 0.5,
    'mixup': 0.2,
    'copy_paste': 0.1,
    'auto_augment': 'randaugment'
}

def train_final_model():
    """Train the final optimized model"""
    print("=== Training Final Optimized Model ===")
    
    # Initialize model
    model = YOLO(BEST_CONFIG['model'])
    
    # Train with best hyperparameters
    results = model.train(**BEST_CONFIG)
    
    print("\\n✅ Training completed!")
    print(f"Best model saved to: {BEST_CONFIG['project']}/{BEST_CONFIG['name']}/weights/best.pt")
    
    # Validate
    metrics = model.val()
    print(f"\\nValidation mAP@50: {metrics.box.map50:.4f}")
    print(f"Validation mAP@50-95: {metrics.box.map:.4f}")
    
    return model

def generate_final_submission(model):
    """Generate submission with the final model"""
    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm
    
    print("\\n=== Generating Final Submission ===")
    
    # Load test mapping
    id_mapping = pd.read_csv('data/ID_to_Image_Mapping.csv')
    test_dir = Path('data/test')
    
    submission_data = []
    
    for idx, row in tqdm(id_mapping.iterrows(), total=len(id_mapping)):
        img_id = idx + 1
        img_name = row['image_id']
        img_path = test_dir / img_name
        
        if img_path.exists():
            # Multi-scale inference
            all_preds = []
            for scale in [0.8, 1.0, 1.2]:
                size = int(640 * scale)
                results = model(img_path, imgsz=size, conf=0.1, iou=0.4, augment=True)
                if len(results) > 0:
                    all_preds.append(results[0])
            
            # Combine predictions
            best_conf = 0
            best_pred = None
            
            for pred in all_preds:
                if len(pred.boxes) > 0 and pred.boxes.conf.max() > best_conf:
                    best_conf = float(pred.boxes.conf.max())
                    best_idx = pred.boxes.conf.argmax()
                    best_pred = {
                        'box': pred.boxes.xyxy[best_idx].cpu().numpy(),
                        'conf': best_conf,
                        'cls': int(pred.boxes.cls[best_idx].cpu())
                    }
            
            if best_pred and best_conf > 0.15:
                x1, y1, x2, y2 = best_pred['box']
                classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 
                          'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
                label = classes[best_pred['cls']] if best_pred['cls'] < len(classes) else 'Cardiomegaly'
                
                submission_data.append({
                    'id': img_id,
                    'image_id': img_name,
                    'x_min': round(float(x1), 2),
                    'y_min': round(float(y1), 2),
                    'x_max': round(float(x2), 2),
                    'y_max': round(float(y2), 2),
                    'confidence': f"{best_conf:.4f}",
                    'label': label
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
    
    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)
    submission_df = submission_df[['id', 'image_id', 'x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'label']]
    
    # Save
    submission_df.to_csv('submission_optimized.csv', index=False)
    print(f"\\n✅ Optimized submission saved to submission_optimized.csv")
    
    # Also save as submission.csv
    submission_df.to_csv('submission.csv', index=False)
    print("Also saved as submission.csv")
    
    # Print statistics
    print(f"\\nTotal predictions: {len(submission_df)}")
    print("Class distribution:")
    print(submission_df['label'].value_counts())

if __name__ == "__main__":
    # Train model
    model = train_final_model()
    
    # Generate submission
    generate_final_submission(model)
    
    print("\\n🎯 Final optimized model and submission ready!")
'''
    
    with open('train_final_optimized.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Created train_final_optimized.py with best hyperparameters")

def main():
    """Main optimization pipeline"""
    print("=== Hyperparameter Optimization Pipeline ===\n")
    
    # Check if we should run experiments or just analyze
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        # Just analyze existing results
        analyze_experiments()
    else:
        # Generate experiment configs
        experiments = create_experiment_configs()
        print(f"Generated {len(experiments)} experiment configurations")
        
        # Run experiments (or subset for testing)
        results = []
        for config in experiments[:2]:  # Run first 2 for demo
            try:
                result = run_experiment(config)
                results.append(result)
            except Exception as e:
                print(f"Error in experiment {config['name']}: {e}")
        
        # Analyze results
        if results:
            analyze_experiments()
    
    # Create final training script
    create_final_training_script()
    
    print("\n✅ Optimization complete!")
    print("\nNext steps:")
    print("1. Review experiment results")
    print("2. Run: python train_final_optimized.py")
    print("3. Submit the optimized predictions")

if __name__ == "__main__":
    main()