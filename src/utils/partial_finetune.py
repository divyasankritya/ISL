#!/usr/bin/env python3
"""
Partial fine-tuning script for the Sign Language Recognition Model.
This script loads weights from an existing model where possible,
freezes layers that loaded correctly, and fine-tunes only the layers that didn't load correctly.
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import FINAL_MODEL_PATH, EPOCHS, BATCH_SIZE
from data_utils.data_preprocessor import get_dataset
from models.cnn_lstm_model import build_cnn_lstm_model

# Set TF to grow GPU memory as needed
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s), enabled memory growth")
    except RuntimeError as e:
        print(f"GPU error: {e}")

def load_and_prepare_model():
    """
    Load the model and its weights where possible.
    Return the model with frozen layers for those that loaded correctly.
    
    Returns:
        model: Model with loaded weights where possible and frozen layers
    """
    # Load datasets
    print("Loading dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test, label_map = get_dataset()
    print(f"Dataset loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Build fresh model
    print(f"Building model for {len(label_map)} classes...")
    model = build_cnn_lstm_model(len(label_map))
    
    # Get original layer names
    original_layers = [layer.name for layer in model.layers]
    print(f"Model built with {len(original_layers)} layers")
    
    # Check if weights file exists
    weights_path = FINAL_MODEL_PATH
    if not os.path.exists(weights_path):
        print(f"Error: Weights file {weights_path} not found.")
        sys.exit(1)
    
    # Try to load weights
    print(f"Loading weights from {weights_path}...")
    try:
        # Get original weights before loading to compare
        original_weights = {}
        for layer in model.layers:
            if layer.weights:
                original_weights[layer.name] = [w.numpy() for w in layer.weights]
        
        # Load weights with skip_mismatch
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print("Weights loaded with skip_mismatch=True")
        
        # Check which layers were loaded vs. which have random weights
        loaded_layers = []
        random_layers = []
        
        for layer in model.layers:
            if layer.name in original_weights and layer.weights:
                # Check if weights changed after loading
                current_weights = [w.numpy() for w in layer.weights]
                changed = False
                
                for i, (orig_w, curr_w) in enumerate(zip(original_weights[layer.name], current_weights)):
                    # Check if the weights are different
                    if not np.array_equal(orig_w, curr_w):
                        changed = True
                        break
                
                if changed:
                    loaded_layers.append(layer.name)
                else:
                    random_layers.append(layer.name)
        
        print(f"Layers with loaded weights: {len(loaded_layers)}")
        print(f"Layers with random weights: {len(random_layers)}")
        
        # Freeze layers that loaded correctly
        for layer in model.layers:
            if layer.name in loaded_layers:
                layer.trainable = False
                print(f"Froze layer: {layer.name}")
            else:
                layer.trainable = True
                print(f"Keeping layer trainable: {layer.name}")
        
        # Recompile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    except Exception as e:
        print(f"Error during weight loading: {str(e)}")
        print("Proceeding with freshly initialized model.")
    
    return model, X_train, y_train, X_val, y_val, X_test, y_test, label_map

def partial_finetune(model, X_train, y_train, X_val, y_val):
    """
    Fine-tune the model for a few epochs.
    
    Args:
        model: Model with some frozen layers
        X_train, y_train: Training data
        X_val, y_val: Validation data
    
    Returns:
        history: Training history
    """
    # Prepare output directory
    finetune_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'finetuned')
    os.makedirs(finetune_dir, exist_ok=True)
    
    # Create callbacks
    callbacks = [
        # Save the best model
        ModelCheckpoint(
            os.path.join(finetune_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Reduce learning rate when plateau is reached
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        # Early stopping
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        )
    ]
    
    # Add mixup augmentation for improved generalization
    def mixup_data(x, y, alpha=0.2):
        """
        Applies mixup augmentation to batch data.
        
        Args:
            x: Input data batch
            y: Target labels batch (one-hot encoded)
            alpha: Mixup hyperparameter
            
        Returns:
            Tuple of (mixed_x, mixed_y)
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.shape[0]
        
        # Shuffle indices
        index = tf.random.shuffle(tf.range(batch_size))
        
        # Mix the data
        mixed_x = lam * x + (1 - lam) * tf.gather(x, index)
        mixed_y = lam * y + (1 - lam) * tf.gather(y, index)
        
        return mixed_x, mixed_y
    
    # Generator for mixup augmentation
    def train_generator(x, y, batch_size):
        """Generate batches with mixup augmentation."""
        n_samples = x.shape[0]
        indices = np.arange(n_samples)
        
        while True:
            np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples - batch_size + 1, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_x = x[batch_indices]
                batch_y = y[batch_indices]
                
                # Apply mixup
                batch_x, batch_y = mixup_data(batch_x, batch_y, alpha=0.2)
                    
                yield batch_x, batch_y
    
    # Set up the generator
    train_gen = train_generator(X_train, y_train, BATCH_SIZE)
    steps_per_epoch = len(X_train) // BATCH_SIZE
    
    # Fine-tune for a short number of epochs
    print(f"\nStarting fine-tuning for up to {EPOCHS//3} epochs with mixup augmentation...")
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val, y_val),
        epochs=EPOCHS//3,  # Use fewer epochs for fine-tuning
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final fine-tuned model
    final_model_path = os.path.join(finetune_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f"Fine-tuned model saved to {final_model_path}")
    
    return history, final_model_path

def evaluate_model(model, X_test, y_test, label_map):
    """
    Evaluate the fine-tuned model on test data.
    
    Args:
        model: Fine-tuned model
        X_test, y_test: Test data
        label_map: Mapping from class indices to labels
    
    Returns:
        test_accuracy: Accuracy on test set
    """
    print("\nEvaluating fine-tuned model on test set...")
    
    # Apply test-time augmentation (TTA)
    def apply_test_augmentation(image):
        """Apply simple augmentations to a single image."""
        augmentations = []
        
        # Original image
        augmentations.append(image)
        
        # Horizontal flip
        augmentations.append(tf.image.flip_left_right(image))
        
        # Brightness variations
        augmentations.append(tf.image.adjust_brightness(image, delta=0.1))
        augmentations.append(tf.image.adjust_brightness(image, delta=-0.1))
        
        # Contrast variations
        augmentations.append(tf.image.adjust_contrast(image, factor=1.1))
        
        return augmentations
    
    print("Performing test-time augmentation...")
    
    # Get predictions with test-time augmentation
    n_samples = X_test.shape[0]
    n_classes = y_test.shape[1]
    y_pred_probs = np.zeros((n_samples, n_classes))
    
    for i in range(n_samples):
        # Get the current sequence
        sequence = X_test[i]
        
        # Apply augmentations to each frame in the sequence
        augmented_sequences = []
        
        # Original sequence
        augmented_sequences.append(sequence)
        
        # Brightness adjusted sequence
        bright_seq = np.clip(sequence * 1.1, 0, 1)
        augmented_sequences.append(bright_seq)
        
        dark_seq = np.clip(sequence * 0.9, 0, 1)
        augmented_sequences.append(dark_seq)
        
        # Make predictions for each augmented sequence
        sequence_preds = []
        for aug_seq in augmented_sequences:
            # Add batch dimension
            batched_seq = np.expand_dims(aug_seq, axis=0)
            # Get prediction
            pred = model.predict(batched_seq, verbose=0)
            sequence_preds.append(pred[0])
        
        # Average the predictions
        avg_pred = np.mean(sequence_preds, axis=0)
        y_pred_probs[i] = avg_pred
    
    # Get predicted classes
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    from sklearn.metrics import accuracy_score
    test_acc = accuracy_score(y_true_classes, y_pred_classes)
    print(f"\nTest accuracy with TTA: {test_acc:.4f}")
    
    # Process label_map
    if isinstance(label_map, dict):
        if all(k.isdigit() for k in label_map.keys()):
            target_names = [label_map[str(i)] for i in range(len(label_map))]
        else:
            inverted_map = {v: k for k, v in label_map.items()}
            target_names = [inverted_map.get(i, f"Class {i}") for i in range(len(label_map))]
    else:
        target_names = [f"Class {i}" for i in range(len(label_map))]
    
    # Classification report
    from sklearn.metrics import classification_report
    unique_classes = sorted(set(np.concatenate([y_true_classes, y_pred_classes])))
    report = classification_report(
        y_true_classes, y_pred_classes, 
        labels=unique_classes,
        target_names=[target_names[i] for i in unique_classes] if len(target_names) >= len(unique_classes) else None,
        output_dict=True
    )
    
    # Save report
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation_results')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, 'finetuned_report.json')
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"Evaluation report saved to {report_path}")
    
    # Print per-class metrics
    print("\nPer-class accuracy:")
    for class_idx in range(len(label_map)):
        mask = (y_true_classes == class_idx)
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred_classes[mask] == class_idx)
            print(f"Class {class_idx} ({target_names[class_idx]}): {class_acc:.4f} ({np.sum(mask)} samples)")
    
    return test_acc

def main():
    print("=" * 80)
    print("PARTIAL FINE-TUNING OF SIGN LANGUAGE RECOGNITION MODEL")
    print("=" * 80)
    
    start_time = time.time()
    
    # Load model and prepare for fine-tuning
    model, X_train, y_train, X_val, y_val, X_test, y_test, label_map = load_and_prepare_model()
    
    # Fine-tune the model
    history, model_path = partial_finetune(model, X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    test_acc = evaluate_model(model, X_test, y_test, label_map)
    
    total_time = time.time() - start_time
    print(f"\nFine-tuning completed in {total_time:.2f} seconds")
    print(f"Final test accuracy: {test_acc:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main() 