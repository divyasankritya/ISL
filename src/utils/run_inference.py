#!/usr/bin/env python3
"""
Inference script for the fine-tuned Sign Language Recognition Model.
This script loads the fine-tuned model and runs inference on test data or video input.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Add sign_language_project directory to path for imports
SIGN_LANGUAGE_PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sign_language_project')
if os.path.exists(SIGN_LANGUAGE_PROJECT_DIR):
    sys.path.append(SIGN_LANGUAGE_PROJECT_DIR)
else:
    # If we're already in the sign_language_project directory
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Define the custom TemporalAttention layer here to use when loading the model
class TemporalAttention(Layer):
    """Custom temporal attention layer to focus on important frames in a sequence."""
    def __init__(self, units, trainable=True, dtype=None, **kwargs):
        super(TemporalAttention, self).__init__(trainable=trainable, dtype=dtype, **kwargs)
        self.units = units
        self.W1 = None
        self.W2 = None
        self.supports_masking = True

    def build(self, input_shape):
        self.W1 = self.add_weight(
            name='W1',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W2 = self.add_weight(
            name='W2',
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(TemporalAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        # inputs shape: (batch_size, time_steps, features)
        # Calculate attention weights
        # W1 shape: (features, units)
        # W2 shape: (units, 1)
        # Output shape: (batch_size, time_steps, 1)
        attention = tf.tanh(tf.matmul(inputs, self.W1))  # (batch_size, time_steps, units)
        attention = tf.matmul(attention, self.W2)  # (batch_size, time_steps, 1)
        
        # Apply masking if provided
        if mask is not None:
            # Add a dimension for the attention logits
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(mask, -1)
            # Add a large negative value to the masked timesteps
            attention = attention * mask + (1.0 - mask) * (-1e9)
        
        # Apply softmax to get attention weights
        attention = tf.nn.softmax(attention, axis=1)  # (batch_size, time_steps, 1)
        
        # Apply attention weights to get context vector
        context = inputs * attention  # Broadcasting: (batch_size, time_steps, features)
        
        return context

    def compute_output_shape(self, input_shape):
        return input_shape
        
    def get_config(self):
        config = {
            'units': self.units
        }
        base_config = super(TemporalAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Now import the modules
from data_utils.data_preprocessor import get_dataset
from config import FINAL_MODEL_PATH

# Set TF to grow GPU memory as needed
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s), enabled memory growth")
    except RuntimeError as e:
        print(f"GPU error: {e}")

def load_model(model_path):
    """
    Load the fine-tuned model from the specified path.
    
    Args:
        model_path: Path to the saved model file (.h5)
        
    Returns:
        The loaded model
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        sys.exit(1)
        
    print(f"Loading model from {model_path}...")
    
    # Load the model with custom objects to handle the TemporalAttention layer
    custom_objects = {'TemporalAttention': TemporalAttention}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print("Model loaded successfully.")
    
    return model

def test_time_augmentation(model, sequence):
    """
    Apply test-time augmentation to improve prediction accuracy.
    
    Args:
        model: The loaded model
        sequence: Input sequence to predict
        
    Returns:
        Average prediction probabilities
    """
    # Create augmented versions of the sequence
    augmented_sequences = []
    
    # Original sequence
    augmented_sequences.append(sequence)
    
    # Brightness adjusted sequences
    bright_seq = np.clip(sequence * 1.1, 0, 1)
    augmented_sequences.append(bright_seq)
    
    dark_seq = np.clip(sequence * 0.9, 0, 1)
    augmented_sequences.append(dark_seq)
    
    # Make predictions for each augmented sequence
    all_preds = []
    for aug_seq in augmented_sequences:
        # Always add batch dimension (None) as first dimension
        if len(aug_seq.shape) == 4:  # (frames, height, width, channels)
            aug_seq = np.expand_dims(aug_seq, axis=0)  # Add batch dimension: (1, frames, height, width, channels)
        # Get prediction
        pred = model.predict(aug_seq, verbose=0)
        all_preds.append(pred[0])  # First element because we have batch size 1
    
    # Average the predictions
    avg_pred = np.mean(all_preds, axis=0)
    return avg_pred

def evaluate_on_test_data(model, label_map):
    """
    Evaluate the model on the test dataset.
    
    Args:
        model: The loaded model
        label_map: Mapping from class indices to labels
        
    Returns:
        test_accuracy: Accuracy on test set
    """
    print("Loading test dataset...")
    _, _, _, _, X_test, y_test, _ = get_dataset()
    print(f"Test data loaded. Shape: {X_test.shape}")
    
    # Get predictions with test-time augmentation
    n_samples = X_test.shape[0]
    n_classes = y_test.shape[1]
    y_pred_probs = np.zeros((n_samples, n_classes))
    
    print("Running inference on test data...")
    for i in range(n_samples):
        # Apply TTA to each sequence
        sequence = X_test[i]
        y_pred_probs[i] = test_time_augmentation(model, sequence)
        
        # Print progress
        if (i+1) % 10 == 0 or i+1 == n_samples:
            print(f"Processed {i+1}/{n_samples} samples")
    
    # Get predicted classes
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    from sklearn.metrics import accuracy_score
    test_acc = accuracy_score(y_true_classes, y_pred_classes)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Process label_map for readable class names
    if isinstance(label_map, dict):
        if all(k.isdigit() for k in label_map.keys()):
            target_names = [label_map[str(i)] for i in range(len(label_map))]
        else:
            inverted_map = {v: k for k, v in label_map.items()}
            target_names = [inverted_map.get(i, f"Class {i}") for i in range(len(label_map))]
    else:
        target_names = [str(label) for label in label_map]
    
    # Print classification report
    unique_classes = sorted(set(np.concatenate([y_true_classes, y_pred_classes])))
    report = classification_report(
        y_true_classes, y_pred_classes, 
        labels=unique_classes,
        target_names=[target_names[i] for i in unique_classes] if len(target_names) >= len(unique_classes) else None,
        output_dict=True
    )
    
    # Save report
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inference_results')
    os.makedirs(results_dir, exist_ok=True)
    
    report_path = os.path.join(results_dir, 'inference_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Print and save confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    cm_display = confusion_matrix(y_true_classes, y_pred_classes, normalize='true')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_display, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    
    # Print per-class metrics
    print("\nPer-class accuracy:")
    for class_idx in range(len(target_names)):
        mask = (y_true_classes == class_idx)
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred_classes[mask] == class_idx)
            print(f"Class {class_idx} ({target_names[class_idx]}): {class_acc:.4f} ({np.sum(mask)} samples)")
    
    print(f"\nDetailed evaluation results saved to {results_dir}")
    return test_acc

def predict_single_sample(model, sample_index, label_map):
    """
    Run inference on a single sample from the test set.
    
    Args:
        model: The loaded model
        sample_index: Index of the sample in test set
        label_map: Mapping from class indices to labels
    """
    # Load test data
    _, _, _, _, X_test, y_test, _ = get_dataset()
    
    if sample_index >= len(X_test):
        print(f"Error: Sample index {sample_index} is out of bounds. Test set has {len(X_test)} samples.")
        return
    
    # Get the sample
    sample = X_test[sample_index]
    true_label_idx = np.argmax(y_test[sample_index])
    
    # Process label_map for readable class names
    if isinstance(label_map, dict):
        if all(k.isdigit() for k in label_map.keys()):
            target_names = [label_map[str(i)] for i in range(len(label_map))]
        else:
            inverted_map = {v: k for k, v in label_map.items()}
            target_names = [inverted_map.get(i, f"Class {i}") for i in range(len(label_map))]
    else:
        target_names = [str(label) for label in label_map]
    
    # Get true label
    true_label = target_names[true_label_idx]
    
    # Predict with TTA
    pred_probs = test_time_augmentation(model, sample)
    pred_label_idx = np.argmax(pred_probs)
    pred_label = target_names[pred_label_idx]
    
    # Print results
    print(f"\nSample {sample_index} Prediction:")
    print(f"True label: {true_label} (class {true_label_idx})")
    print(f"Predicted label: {pred_label} (class {pred_label_idx})")
    print(f"Confidence: {pred_probs[pred_label_idx]:.4f}")
    
    # Print top 3 predictions
    top_indices = np.argsort(pred_probs)[::-1][:3]
    print("\nTop 3 predictions:")
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {target_names[idx]}: {pred_probs[idx]:.4f}")
    
    # Visualize the sequence frames
    if len(sample.shape) == 4:  # (frames, height, width, channels)
        n_frames = min(5, sample.shape[0])  # Display up to 5 frames
        plt.figure(figsize=(15, 3))
        for i in range(n_frames):
            plt.subplot(1, n_frames, i+1)
            frame_idx = i * sample.shape[0] // n_frames
            plt.imshow(sample[frame_idx], cmap='gray')
            plt.title(f"Frame {frame_idx}")
            plt.axis('off')
        
        plt.suptitle(f"True: {true_label} | Predicted: {pred_label}")
        plt.tight_layout()
        
        # Save the visualization
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inference_results')
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, f'sample_{sample_index}_prediction.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Run inference with fine-tuned Sign Language Recognition model')
    
    # Model path argument
    parser.add_argument('--model', type=str, 
                      default='/Users/nilopakumar/Desktop/ISL_NEW/sign_language_project/models/finetuned/final_model.h5',
                      help='Path to the fine-tuned model')
    
    # Evaluation mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--evaluate', action='store_true',
                     help='Evaluate model on the test dataset')
    group.add_argument('--sample', type=int, 
                     help='Run inference on a specific sample from test set (specify index)')
    
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model)
    
    # Get class labels from the dataset
    _, _, _, _, _, _, label_map = get_dataset()
    
    # Determine the action based on arguments
    if args.evaluate:
        print("Evaluating model on test dataset...")
        test_acc = evaluate_on_test_data(model, label_map)
        print(f"Final test accuracy: {test_acc:.4f}")
    
    elif args.sample is not None:
        print(f"Running inference on sample {args.sample}...")
        predict_single_sample(model, args.sample, label_map)
    
if __name__ == "__main__":
    main() 