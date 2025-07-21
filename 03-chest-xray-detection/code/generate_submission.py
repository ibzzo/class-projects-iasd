#!/usr/bin/env python3
"""
Generate submission.csv using pre-trained model
"""

import os
import pandas as pd
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from tqdm import tqdm

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'final_model.pth'  # or 'best_model.pth'
TEST_DIR = 'data/test'
ID_MAPPING_PATH = 'data/ID_to_Image_Mapping.csv'
IMG_SIZE = 512
CONFIDENCE_THRESHOLD = 0.3

# Classes mapping
PATHOLOGY_CLASSES = {
    'Cardiomegaly': 1,
    'Effusion': 2,
    'Infiltrate': 3,
    'Mass': 4,
    'Nodule': 5,
    'Pneumonia': 6,
    'Pneumothorax': 7,
    'Atelectasis': 8
}
NUM_CLASSES = len(PATHOLOGY_CLASSES) + 1

print(f"Using device: {DEVICE}")
print(f"Loading model from: {MODEL_PATH}")

# Initialize model
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Load model
model = get_model(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Transform
transform = transforms.ToTensor()

# Inverse label mapping
id_to_label = {v: k for k, v in PATHOLOGY_CLASSES.items()}

# Load ID mapping
id_mapping = pd.read_csv(ID_MAPPING_PATH)
print(f"\nTest images to process: {len(id_mapping)}")

# Generate predictions
predictions = []

with torch.no_grad():
    for _, row in tqdm(id_mapping.iterrows(), total=len(id_mapping), desc='Generating predictions'):
        img_id = row['id']
        img_name = row['image_id']
        img_path = os.path.join(TEST_DIR, img_name)
        
        if os.path.exists(img_path):
            try:
                # Load and preprocess image
                image = Image.open(img_path).convert('RGB')
                orig_size = image.size[0]  # Assuming square images
                image_resized = image.resize((IMG_SIZE, IMG_SIZE))
                image_tensor = transform(image_resized).unsqueeze(0).to(DEVICE)
                
                # Get predictions
                outputs = model(image_tensor)[0]
                
                # Scale factor to original size
                scale = orig_size / IMG_SIZE
                
                # Process predictions
                pred_added = False
                
                if len(outputs['boxes']) > 0:
                    # Sort by confidence and take best predictions
                    scores = outputs['scores'].cpu().numpy()
                    sorted_indices = scores.argsort()[::-1]
                    
                    for idx in sorted_indices:
                        score = float(scores[idx])
                        if score < CONFIDENCE_THRESHOLD:
                            break
                            
                        box = outputs['boxes'][idx].cpu().numpy()
                        label = int(outputs['labels'][idx].cpu().item())
                        
                        # Scale box coordinates
                        x1, y1, x2, y2 = box * scale
                        
                        # Ensure valid coordinates
                        x1 = max(0, min(x1, orig_size))
                        y1 = max(0, min(y1, orig_size))
                        x2 = max(x1, min(x2, orig_size))
                        y2 = max(y1, min(y2, orig_size))
                        
                        predictions.append({
                            'id': int(img_id),
                            'image_id': str(img_name),
                            'x_min': round(float(x1), 2),
                            'y_min': round(float(y1), 2),
                            'x_max': round(float(x2), 2),
                            'y_max': round(float(y2), 2),
                            'confidence': round(score, 4),
                            'label': id_to_label.get(label, 'Cardiomegaly')
                        })
                        pred_added = True
                
                # Add default prediction if none found
                if not pred_added:
                    predictions.append({
                        'id': int(img_id),
                        'image_id': str(img_name),
                        'x_min': 300.0,
                        'y_min': 300.0,
                        'x_max': 700.0,
                        'y_max': 700.0,
                        'confidence': 0.1,
                        'label': 'Cardiomegaly'
                    })
                    
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                # Default prediction on error
                predictions.append({
                    'id': int(img_id),
                    'image_id': str(img_name),
                    'x_min': 300.0,
                    'y_min': 300.0,
                    'x_max': 700.0,
                    'y_max': 700.0,
                    'confidence': 0.1,
                    'label': 'Cardiomegaly'
                })
        else:
            print(f"Warning: {img_name} not found")
            # Default prediction if file not found
            predictions.append({
                'id': int(img_id),
                'image_id': str(img_name),
                'x_min': 300.0,
                'y_min': 300.0,
                'x_max': 700.0,
                'y_max': 700.0,
                'confidence': 0.1,
                'label': 'Cardiomegaly'
            })

# Create submission dataframe
submission_df = pd.DataFrame(predictions)

# Ensure correct column order and types
submission_df['id'] = submission_df['id'].astype(int)
submission_df['image_id'] = submission_df['image_id'].astype(str)
submission_df['x_min'] = submission_df['x_min'].astype(float)
submission_df['y_min'] = submission_df['y_min'].astype(float)
submission_df['x_max'] = submission_df['x_max'].astype(float)
submission_df['y_max'] = submission_df['y_max'].astype(float)
submission_df['confidence'] = submission_df['confidence'].astype(float)
submission_df['label'] = submission_df['label'].astype(str)

# Reorder columns to match required format
submission_df = submission_df[['id', 'image_id', 'x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'label']]

# Verify we have predictions for all images
print(f"\nTotal predictions: {len(submission_df)}")
print(f"Unique images with predictions: {submission_df['id'].nunique()}")
print(f"Expected images: {len(id_mapping)}")

# Check if we're missing any images
predicted_ids = set(submission_df['id'].unique())
expected_ids = set(id_mapping['id'].values)
missing_ids = expected_ids - predicted_ids

if missing_ids:
    print(f"Warning: Missing predictions for {len(missing_ids)} images")
    for missing_id in missing_ids:
        missing_row = id_mapping[id_mapping['id'] == missing_id].iloc[0]
        submission_df = pd.concat([submission_df, pd.DataFrame([{
            'id': int(missing_id),
            'image_id': str(missing_row['image_id']),
            'x_min': 300.0,
            'y_min': 300.0,
            'x_max': 700.0,
            'y_max': 700.0,
            'confidence': 0.1,
            'label': 'Cardiomegaly'
        }])], ignore_index=True)

# Save submission
submission_df.to_csv('submission.csv', index=False)

print(f"\n✅ Submission saved to submission.csv")
print(f"Final total predictions: {len(submission_df)}")
print(f"Unique images: {submission_df['id'].nunique()}")
print(f"\nPrediction distribution:")
print(submission_df['label'].value_counts())
print(f"\nFirst 5 rows:")
print(submission_df.head())
print(f"\nSample with multiple detections (if any):")
# Show an example with multiple detections
multi_detection = submission_df.groupby('id').size().sort_values(ascending=False).head(1)
if len(multi_detection) > 0 and multi_detection.iloc[0] > 1:
    sample_id = multi_detection.index[0]
    print(submission_df[submission_df['id'] == sample_id])