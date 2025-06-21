"""
Training script for the Sign Language Recognition Model.
This script handles the training and validation of the CNN+LSTM model.
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    EPOCHS, BATCH_SIZE, CHECKPOINT_PATH, 
    FINAL_MODEL_PATH, HISTORY_PATH, LABEL_MAP_PATH
)
from models.cnn_lstm_model import build_cnn_lstm_model, get_callbacks
from data_utils.data_preprocessor import get_dataset

# Set memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s), enabled memory growth")
    except RuntimeError as e:
        print(f"GPU error: {e}")

# Custom mixup data augmentation
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

# Custom callback for gradient accumulation
class GradientAccumulationCallback(Callback):
    """
    Callback that implements gradient accumulation to simulate larger batch sizes.
    Compatible with mixed precision training.
    """
    def __init__(self, accumulation_steps=4):
        super(GradientAccumulationCallback, self).__init__()
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.gradient_accumulation = False
        
    def on_train_begin(self, logs=None):
        # Check if we can properly access gradients with this optimizer
        # This is needed for compatibility with mixed precision training
        try:
            # For mixed precision, we need to use the unwrapped optimizer
            if hasattr(self.model.optimizer, 'inner_optimizer'):
                optimizer = self.model.optimizer.inner_optimizer
            else:
                optimizer = self.model.optimizer
                
            # Test if we can get gradients
            self.gradient_accumulation = True
            print("Gradient accumulation is enabled.")
        except (AttributeError, NotImplementedError):
            # If we can't get gradients, disable gradient accumulation
            self.gradient_accumulation = False
            print("Gradient accumulation is disabled due to optimizer incompatibility.")
    
    def on_train_batch_begin(self, batch, logs=None):
        # Skip if gradient accumulation is disabled
        if not self.gradient_accumulation:
            return
            
        # Start a new accumulation cycle
        if self.current_step % self.accumulation_steps == 0:
            # Use the model subclassing API to control gradients
            self.model._accumulate_next_grad = True
    
    def on_train_batch_end(self, batch, logs=None):
        # Skip if gradient accumulation is disabled
        if not self.gradient_accumulation:
            return
            
        # Increment step counter
        self.current_step += 1
        
        # Apply accumulated gradients at the end of accumulation cycle
        if self.current_step % self.accumulation_steps == 0:
            # Use the model subclassing API to apply gradients
            self.model._apply_accumulated_grad = True

# Learning rate finder callback
class LRFinder(Callback):
    """
    Callback to find an optimal learning rate for training.
    """
    def __init__(self, min_lr=1e-7, max_lr=1e-2, n_steps=100, beta=0.98):
        super(LRFinder, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.n_steps = n_steps
        self.beta = beta
        self.step = 0
        self.best_loss = float('inf')
        self.lrs = []
        self.losses = []
        self.smoothed_losses = []
        
    def on_train_begin(self, logs=None):
        self.step = 0
        self.best_loss = float('inf')
        self.lrs = []
        self.losses = []
        self.smoothed_losses = []
        
    def on_train_batch_end(self, batch, logs=None):
        # Calculate the learning rate
        lr = self.min_lr * (self.max_lr / self.min_lr) ** (self.step / self.n_steps)
        
        # Set the learning rate
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        
        # Record the learning rate and loss
        self.lrs.append(lr)
        current_loss = logs.get('loss')
        self.losses.append(current_loss)
        
        # Smooth the loss
        if self.step == 0:
            smoothed_loss = current_loss
        else:
            smoothed_loss = self.beta * self.smoothed_losses[-1] + (1 - self.beta) * current_loss
            # Correct the bias
            smoothed_loss = smoothed_loss / (1 - self.beta ** (self.step + 1))
            
        self.smoothed_losses.append(smoothed_loss)
        
        # Check if the loss is diverging
        if self.step > 0 and smoothed_loss > 4 * self.best_loss:
            self.model.stop_training = True
            return
            
        if smoothed_loss < self.best_loss:
            self.best_loss = smoothed_loss
            
        self.step += 1
        
        # Stop after n_steps
        if self.step >= self.n_steps:
            self.model.stop_training = True
            
    def plot(self, save_path=None):
        """Plot the learning rate finder results."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.lrs, self.smoothed_losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        
        # Find the optimal learning rate
        min_grad_idx = np.gradient(np.array(self.smoothed_losses)).argmin()
        optimal_lr = self.lrs[min_grad_idx]
        plt.axvline(optimal_lr, color='r', linestyle='--', 
                   label=f'Optimal LR: {optimal_lr:.1e}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
        
        return optimal_lr

def find_optimal_lr():
    """
    Find the optimal learning rate for training.
    
    Returns:
        Optimal learning rate
    """
    print("Finding optimal learning rate...")
    X_train, y_train, X_val, y_val, _, _, label_map = get_dataset()
    
    # Build a small model for LR finding
    model = build_cnn_lstm_model(len(label_map))
    
    # Initialize LR finder
    lr_finder = LRFinder(min_lr=1e-7, max_lr=1e-2, n_steps=50)
    
    # Train for a few steps with the LR finder
    model.fit(
        X_train[:100], y_train[:100],
        epochs=1,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[lr_finder]
    )
    
    # Plot and get optimal LR
    lr_plot_path = os.path.join(os.path.dirname(HISTORY_PATH), 'lr_finder.png')
    optimal_lr = lr_finder.plot(save_path=lr_plot_path)
    
    # Return optimal LR (divided by 10 for safety)
    return optimal_lr / 10

# Simple gradient accumulation model wrapper
class GradientAccumulationModel(tf.keras.Model):
    """
    Wrapper around keras Model to implement gradient accumulation
    without relying on optimizer internals.
    """
    def __init__(self, model, accumulation_steps=4):
        super(GradientAccumulationModel, self).__init__()
        self.model = model
        self.accumulation_steps = accumulation_steps
        self._accumulate_next_grad = False
        self._apply_accumulated_grad = False
        self._accumulated_gradients = None
        self._accumulation_count = 0
    
    def call(self, inputs, training=None):
        return self.model(inputs, training=training)
    
    def train_step(self, data):
        # Unpack the data
        x, y = data
        
        # Run forward pass
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        
        # Either accumulate gradients or apply them
        if self._accumulate_next_grad:
            if self._accumulated_gradients is None:
                self._accumulated_gradients = [tf.zeros_like(g) for g in gradients]
            
            for i, g in enumerate(gradients):
                self._accumulated_gradients[i] += g
            
            self._accumulation_count += 1
            self._accumulate_next_grad = False
        
        # Apply accumulated gradients
        if self._apply_accumulated_grad and self._accumulated_gradients is not None:
            # Scale the gradients by accumulation steps
            scaled_gradients = [g / self._accumulation_count for g in self._accumulated_gradients]
            
            # Apply gradients
            self.optimizer.apply_gradients(zip(scaled_gradients, self.trainable_variables))
            
            # Reset accumulation
            self._accumulated_gradients = None
            self._accumulation_count = 0
            self._apply_accumulated_grad = False
        
        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

def train_model(use_mixed_precision=True, use_gradient_accumulation=True, find_lr=False, 
                optimal_lr=None, use_mixup=True):
    """
    Train the CNN+LSTM model on the preprocessed dataset with advanced techniques.
    
    Args:
        use_mixed_precision: Whether to use mixed precision training
        use_gradient_accumulation: Whether to use gradient accumulation
        find_lr: Whether to find the optimal learning rate
        optimal_lr: Optimal learning rate to use (if find_lr is False)
        use_mixup: Whether to use mixup augmentation
        
    Returns:
        Trained model and training history
    """
    # Enable mixed precision if requested
    if use_mixed_precision:
        print("Enabling mixed precision training...")
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    
    # Find optimal learning rate if requested
    if find_lr:
        optimal_lr = find_optimal_lr()
        print(f"Found optimal learning rate: {optimal_lr:.1e}")
    elif optimal_lr is None:
        # Use a sensible default
        optimal_lr = 5e-4
    
    print("Loading dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test, label_map = get_dataset()
    num_classes = len(label_map)
    
    print(f"Building model for {num_classes} classes...")
    model = build_cnn_lstm_model(num_classes)
    model.summary()
    
    # Set the learning rate
    model.optimizer.learning_rate.assign(optimal_lr)
    
    # Wrap model with gradient accumulation if requested
    if use_gradient_accumulation and not use_mixed_precision:
        # Note: We only use gradient accumulation if mixed precision is OFF
        # to avoid compatibility issues
        print("Using gradient accumulation with custom model wrapper...")
        model = GradientAccumulationModel(model, accumulation_steps=4)
        model.compile(
            optimizer=model.model.optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # Create callbacks
    print("Setting up training callbacks...")
    callbacks = get_callbacks(CHECKPOINT_PATH)
    
    # Create training data generator for mixup if requested
    def train_generator(x, y, batch_size):
        """Generate batches of training data with mixup augmentation."""
        n_samples = x.shape[0]
        indices = np.arange(n_samples)
        
        while True:
            np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples - batch_size + 1, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_x = x[batch_indices]
                batch_y = y[batch_indices]
                
                if use_mixup:
                    batch_x, batch_y = mixup_data(batch_x, batch_y)
                    
                yield batch_x, batch_y
    
    # Train the model
    print(f"Starting training for {EPOCHS} epochs with learning rate {optimal_lr:.1e}...")
    start_time = time.time()
    
    if use_mixup:
        print("Using mixup augmentation...")
        train_gen = train_generator(X_train, y_train, BATCH_SIZE)
        steps_per_epoch = len(X_train) // BATCH_SIZE
        
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save the final model - unwrap if using gradient accumulation
    if use_gradient_accumulation and not use_mixed_precision:
        model.model.save(FINAL_MODEL_PATH)
    else:
        model.save(FINAL_MODEL_PATH)
    print(f"Model saved to {FINAL_MODEL_PATH}")
    
    # Save training history
    with open(HISTORY_PATH, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_dict = history.history.copy()
        for key in history_dict:
            history_dict[key] = [float(val) for val in history_dict[key]]
        json.dump(history_dict, f, indent=4)
    
    print(f"Training history saved to {HISTORY_PATH}")
    
    return model, history

def plot_training_history(history):
    """
    Plot and save the training history (accuracy and loss curves).
    
    Args:
        history: Training history object from model.fit()
    """
    plt.figure(figsize=(15, 6))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot learning rate if available
    if 'lr' in history.history:
        plt.subplot(1, 3, 3)
        plt.plot(history.history['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
    
    # Save the plot
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(HISTORY_PATH), 'training_history.png')
    plt.savefig(plot_path)
    print(f"Training history plot saved to {plot_path}")
    plt.close()

def evaluate_model(model, X_test, y_test, label_map):
    """
    Evaluate the trained model on the test set.
    
    Args:
        model: Trained Keras model
        X_test: Test data
        y_test: Test labels
        label_map: Mapping from class indices to sentence labels
    
    Returns:
        Test accuracy and per-class metrics
    """
    print("Evaluating model on test set...")
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Get predictions with test-time augmentation
    n_tta = 5  # Number of test-time augmentations
    y_pred = np.zeros((X_test.shape[0], len(label_map)))
    
    for i in range(n_tta):
        # Apply random brightness/contrast augmentation
        X_test_aug = X_test.copy()
        if i > 0:  # Skip augmentation for the first iteration
            # Apply simple brightness/contrast augmentation
            brightness_factor = np.random.uniform(0.9, 1.1)
            X_test_aug = np.clip(X_test_aug * brightness_factor, 0.0, 1.0)
        
        # Get predictions for this augmentation
        y_pred_i = model.predict(X_test_aug)
        y_pred += y_pred_i
    
    # Average predictions
    y_pred /= n_tta
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate per-class accuracy
    class_metrics = {}
    for class_idx, class_name in label_map.items():
        class_idx = int(class_idx)  # Convert string indices to int if necessary
        mask = (y_true_classes == class_idx)
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred_classes[mask] == class_idx)
            class_metrics[class_name] = float(class_acc)
    
    # Save metrics
    metrics = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'per_class_accuracy': class_metrics
    }
    
    metrics_path = os.path.join(os.path.dirname(HISTORY_PATH), 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Test metrics saved to {metrics_path}")
    return test_acc, metrics

if __name__ == "__main__":
    # Train the model with advanced techniques
    model, history = train_model(
        use_mixed_precision=True,
        use_gradient_accumulation=False,  # Disable by default when mixed precision is enabled
        find_lr=True,
        use_mixup=True
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    X_train, y_train, X_val, y_val, X_test, y_test, label_map = get_dataset()
    evaluate_model(model, X_test, y_test, label_map)
    
    print("Training and evaluation complete!") 