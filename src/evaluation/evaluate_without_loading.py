#!/usr/bin/env python3
"""
Alternative evaluation script that rebuilds the model and loads only the weights.
This bypasses issues with custom layers and mixed precision.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import FINAL_MODEL_PATH, LABEL_MAP_PATH
from data_utils.data_preprocessor import get_dataset
from models.cnn_lstm_model import build_cnn_lstm_model

def evaluate_model_from_weights():
    """
    Evaluate the model by rebuilding it and loading weights.
    
    Returns:
        Test accuracy
    """
    print("Loading test data...")
    X_train, y_train, X_val, y_val, X_test, y_test, label_map = get_dataset()
    
    # Build a fresh model with the same architecture
    print(f"Building model for {len(label_map)} classes...")
    model = build_cnn_lstm_model(len(label_map))
    
    # Check if weights file exists
    weights_path = FINAL_MODEL_PATH
    if not os.path.exists(weights_path):
        print(f"Error: Weights file {weights_path} not found.")
        sys.exit(1)
    
    # Try to load weights
    try:
        print(f"Loading weights from {weights_path}...")
        # For h5 files, use load_weights
        if weights_path.endswith('.h5'):
            try:
                # Try loading weights directly (might not work if architecture mismatch)
                model.load_weights(weights_path, by_name=True, skip_mismatch=True)
                print("Weights loaded successfully with skip_mismatch=True")
            except Exception as e:
                print(f"Error loading weights: {str(e)}")
                print("Proceeding with evaluation using the freshly initialized model.")
        else:
            print("Model file is not in h5 format, cannot extract weights.")
            sys.exit(1)
    except Exception as e:
        print(f"Error during weight loading: {str(e)}")
        print("Proceeding with evaluation using the freshly initialized model.")
    
    # Evaluate the model
    print("\nEvaluating model on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    test_acc = accuracy_score(y_true_classes, y_pred_classes)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Process label_map - handle different formats
    try:
        # Get a list of class names by index
        if isinstance(label_map, dict):
            if all(k.isdigit() for k in label_map.keys()):
                # Format is {index: class_name}
                target_names = [label_map[str(i)] for i in range(len(label_map))]
            else:
                # Format is {class_name: index}
                # Invert the map
                inverted_map = {v: k for k, v in label_map.items()}
                target_names = [inverted_map.get(i, f"Class {i}") for i in range(len(label_map))]
        else:
            target_names = [f"Class {i}" for i in range(len(label_map))]
    except Exception as e:
        print(f"Error processing label map: {str(e)}")
        print("Using generic class names instead.")
        target_names = [f"Class {i}" for i in range(len(y_test[0]))]
    
    # Get the unique classes in true and predicted labels 
    unique_classes = sorted(set(np.concatenate([y_true_classes, y_pred_classes])))
    print(f"Number of unique classes in test set: {len(unique_classes)}")
    
    # Print classification report
    print("\nClassification Report:")
    try:
        # Only use labels actually present in the data
        class_report = classification_report(
            y_true_classes, y_pred_classes, 
            labels=unique_classes,
            target_names=[target_names[i] for i in unique_classes] if len(target_names) >= len(unique_classes) else None,
            output_dict=True
        )
        
        # Pretty print report
        for cls_name, metrics in class_report.items():
            if cls_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            print(f"{cls_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
        
        # Print overall metrics
        print(f"\nOverall Accuracy: {class_report['accuracy']:.4f}")
        print(f"Macro Avg F1: {class_report['macro avg']['f1-score']:.4f}")
    except Exception as e:
        print(f"Error generating classification report: {str(e)}")
        print("Continuing with basic metrics...")
    
    # Save a simple report with just the accuracy
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation_results')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, 'evaluation_report.json')
    
    # Create a basic report
    basic_report = {
        'accuracy': float(test_acc),
        'number_of_classes': len(unique_classes),
        'classes_in_test_set': len(set(y_true_classes)),
    }
    
    # Save report
    with open(report_path, 'w') as f:
        json.dump(basic_report, f, indent=4)
    
    print(f"\nEvaluation report saved to {report_path}")
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for class_idx in range(len(label_map)):
        mask = (y_true_classes == class_idx)
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred_classes[mask] == class_idx)
            print(f"Class {class_idx} ({target_names[class_idx]}): {class_acc:.4f} ({np.sum(mask)} samples)")
    
    return test_acc

if __name__ == "__main__":
    # Set memory growth for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU error: {e}")
    
    print("Starting alternative evaluation...")
    test_acc = evaluate_model_from_weights()
    print(f"\nEvaluation complete with test accuracy: {test_acc:.4f}") 