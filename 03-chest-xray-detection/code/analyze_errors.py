#!/usr/bin/env python3
"""
Analyze model errors and generate insights for improvement
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from collections import defaultdict
import json

def analyze_submission(submission_path='submission.csv'):
    """Analyze submission file for patterns"""
    print("=== Submission Analysis ===\n")
    
    df = pd.read_csv(submission_path)
    
    # Basic statistics
    print(f"Total predictions: {len(df)}")
    print(f"Unique images: {df['image_id'].nunique()}")
    
    # Class distribution
    print("\nClass distribution:")
    class_dist = df['label'].value_counts()
    print(class_dist)
    
    # No Finding ratio
    no_finding_ratio = (df['label'] == 'No Finding').sum() / len(df)
    print(f"\nNo Finding ratio: {no_finding_ratio:.2%}")
    
    # Confidence statistics by class
    print("\nConfidence statistics by class:")
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        if label != 'No Finding':
            confidences = label_df['confidence'].astype(float)
            print(f"{label:15s}: mean={confidences.mean():.3f}, std={confidences.std():.3f}")
    
    # Box size analysis
    print("\nBox size analysis:")
    valid_boxes = df[df['label'] != 'No Finding'].copy()
    valid_boxes['width'] = valid_boxes['x_max'] - valid_boxes['x_min']
    valid_boxes['height'] = valid_boxes['y_max'] - valid_boxes['y_min']
    valid_boxes['area'] = valid_boxes['width'] * valid_boxes['height']
    valid_boxes['aspect_ratio'] = valid_boxes['width'] / valid_boxes['height']
    
    print(f"Average box area: {valid_boxes['area'].mean():.0f} pixels²")
    print(f"Average aspect ratio: {valid_boxes['aspect_ratio'].mean():.2f}")
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Class distribution
    class_dist.plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Class Distribution')
    axes[0, 0].set_ylabel('Count')
    
    # Confidence distribution
    valid_boxes['confidence'].astype(float).hist(bins=30, ax=axes[0, 1])
    axes[0, 1].set_title('Confidence Distribution')
    axes[0, 1].set_xlabel('Confidence')
    
    # Box area distribution
    valid_boxes['area'].hist(bins=30, ax=axes[1, 0])
    axes[1, 0].set_title('Box Area Distribution')
    axes[1, 0].set_xlabel('Area (pixels²)')
    
    # Confidence vs Area scatter
    axes[1, 1].scatter(valid_boxes['area'], valid_boxes['confidence'].astype(float), alpha=0.5)
    axes[1, 1].set_xlabel('Box Area')
    axes[1, 1].set_ylabel('Confidence')
    axes[1, 1].set_title('Confidence vs Box Area')
    
    plt.tight_layout()
    plt.savefig('submission_analysis.png', dpi=150)
    print("\nAnalysis plots saved to submission_analysis.png")
    
    return df

def compare_with_training_data():
    """Compare submission with training data distribution"""
    print("\n=== Training vs Submission Comparison ===\n")
    
    # Load training data
    train_df = pd.read_csv('data/train.csv')
    train_df.columns = ['image', 'label', 'x', 'y', 'w', 'h']
    
    # Load submission
    sub_df = pd.read_csv('submission.csv')
    
    # Compare class distributions
    train_dist = train_df['label'].value_counts(normalize=True)
    sub_dist = sub_df[sub_df['label'] != 'No Finding']['label'].value_counts(normalize=True)
    
    print("Class distribution comparison (%):")
    print(f"{'Class':15s} {'Training':>10s} {'Submission':>12s} {'Difference':>12s}")
    print("-" * 50)
    
    for cls in train_dist.index:
        train_pct = train_dist.get(cls, 0) * 100
        sub_pct = sub_dist.get(cls, 0) * 100
        diff = sub_pct - train_pct
        print(f"{cls:15s} {train_pct:10.1f}% {sub_pct:12.1f}% {diff:+12.1f}%")
    
    # Box size comparison
    train_df['area'] = train_df['w'] * train_df['h']
    sub_valid = sub_df[sub_df['label'] != 'No Finding'].copy()
    sub_valid['area'] = (sub_valid['x_max'] - sub_valid['x_min']) * (sub_valid['y_max'] - sub_valid['y_min'])
    
    print(f"\nBox area comparison:")
    print(f"Training mean: {train_df['area'].mean():.0f} pixels²")
    print(f"Submission mean: {sub_valid['area'].mean():.0f} pixels²")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Class distribution comparison
    x = np.arange(len(train_dist))
    width = 0.35
    ax1.bar(x - width/2, train_dist.values * 100, width, label='Training')
    ax1.bar(x + width/2, [sub_dist.get(cls, 0) * 100 for cls in train_dist.index], width, label='Submission')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Percentage')
    ax1.set_title('Class Distribution: Training vs Submission')
    ax1.set_xticks(x)
    ax1.set_xticklabels(train_dist.index, rotation=45)
    ax1.legend()
    
    # Box size distribution
    ax2.hist(train_df['area'], bins=30, alpha=0.5, label='Training', density=True)
    ax2.hist(sub_valid['area'], bins=30, alpha=0.5, label='Submission', density=True)
    ax2.set_xlabel('Box Area (pixels²)')
    ax2.set_ylabel('Density')
    ax2.set_title('Box Size Distribution')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_vs_submission.png', dpi=150)
    print("\nComparison plots saved to training_vs_submission.png")

def generate_improvement_suggestions():
    """Generate specific suggestions for improvement"""
    print("\n=== Improvement Suggestions ===\n")
    
    # Load submission
    df = pd.read_csv('submission.csv')
    
    suggestions = []
    
    # Check No Finding ratio
    no_finding_ratio = (df['label'] == 'No Finding').sum() / len(df)
    if no_finding_ratio > 0.4:
        suggestions.append({
            'issue': 'High No Finding ratio',
            'impact': 'Missing many detections',
            'solutions': [
                'Lower confidence threshold to 0.1 or 0.05',
                'Use test-time augmentation',
                'Train with more epochs',
                'Try larger model (YOLOv8x)'
            ]
        })
    
    # Check confidence levels
    valid_conf = df[df['label'] != 'No Finding']['confidence'].astype(float)
    if valid_conf.mean() < 0.3:
        suggestions.append({
            'issue': 'Low average confidence',
            'impact': 'Model is uncertain',
            'solutions': [
                'Increase training epochs',
                'Use label smoothing',
                'Add more augmentation',
                'Try ensemble of models'
            ]
        })
    
    # Check class imbalance in predictions
    class_dist = df[df['label'] != 'No Finding']['label'].value_counts()
    if len(class_dist) < 5:
        suggestions.append({
            'issue': 'Limited class diversity',
            'impact': 'Missing rare classes',
            'solutions': [
                'Use class weights in training',
                'Oversample rare classes',
                'Lower confidence threshold for rare classes',
                'Train separate models for rare classes'
            ]
        })
    
    # Check box sizes
    valid_boxes = df[df['label'] != 'No Finding'].copy()
    valid_boxes['area'] = (valid_boxes['x_max'] - valid_boxes['x_min']) * (valid_boxes['y_max'] - valid_boxes['y_min'])
    
    if valid_boxes['area'].std() / valid_boxes['area'].mean() > 1.5:
        suggestions.append({
            'issue': 'High variance in box sizes',
            'impact': 'Model struggles with multi-scale objects',
            'solutions': [
                'Use multi-scale training',
                'Add FPN (Feature Pyramid Network)',
                'Train at higher resolution (1024px)',
                'Use anchor optimization'
            ]
        })
    
    # Print suggestions
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion['issue']}")
        print(f"   Impact: {suggestion['impact']}")
        print("   Solutions:")
        for solution in suggestion['solutions']:
            print(f"   - {solution}")
    
    # Save suggestions to file
    with open('improvement_suggestions.json', 'w') as f:
        json.dump(suggestions, f, indent=2)
    print("\nSuggestions saved to improvement_suggestions.json")
    
    return suggestions

def visualize_predictions(n_samples=10):
    """Visualize sample predictions"""
    print(f"\n=== Visualizing {n_samples} Sample Predictions ===\n")
    
    # Load submission
    df = pd.read_csv('submission.csv')
    
    # Get samples with predictions
    samples = df[df['label'] != 'No Finding'].sample(min(n_samples, len(df[df['label'] != 'No Finding'])))
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    for idx, (_, row) in enumerate(samples.iterrows()):
        if idx >= 10:
            break
            
        # Load image
        img_path = f"data/test/{row['image_id']}"
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Draw box
            x1, y1, x2, y2 = int(row['x_min']), int(row['y_min']), int(row['x_max']), int(row['y_max'])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Add label
            label_text = f"{row['label']} ({float(row['confidence']):.2f})"
            cv2.putText(img, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Show
            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(f"Image {row['id']}")
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150)
    print("Sample predictions saved to sample_predictions.png")

def main():
    """Run all analyses"""
    
    # 1. Analyze submission
    df = analyze_submission()
    
    # 2. Compare with training data
    compare_with_training_data()
    
    # 3. Generate improvement suggestions
    suggestions = generate_improvement_suggestions()
    
    # 4. Visualize predictions
    if os.path.exists('data/test'):
        visualize_predictions()
    
    print("\n=== Analysis Complete ===")
    print("\nGenerated files:")
    print("- submission_analysis.png")
    print("- training_vs_submission.png")
    print("- improvement_suggestions.json")
    print("- sample_predictions.png")
    
    print("\n🎯 Top Priority Actions:")
    if suggestions:
        for solution in suggestions[0]['solutions'][:3]:
            print(f"   ✓ {solution}")

if __name__ == "__main__":
    import os
    os.system("pip install opencv-python matplotlib seaborn -q")
    main()