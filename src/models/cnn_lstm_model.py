"""
CNN+LSTM model for Sign Language Recognition.
This module defines the model architecture and training procedures.
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, LSTM, BatchNormalization, Input, Attention,
    AveragePooling2D, Add, LeakyReLU, GlobalAveragePooling2D,
    Bidirectional, SpatialDropout2D, Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping,
    TensorBoard
)
from tensorflow.keras.regularizers import l2

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    IMAGE_SIZE, MAX_SEQUENCE_LENGTH, LSTM_UNITS, DROPOUT_RATE,
    CNN_FILTERS, CNN_KERNEL_SIZE, CNN_POOL_SIZE, LEARNING_RATE
)

class TemporalAttention(Layer):
    """Custom temporal attention layer to focus on important frames in a sequence."""
    def __init__(self, units):
        super(TemporalAttention, self).__init__()
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        
        # Calculate attention scores
        score = self.W2(tf.nn.tanh(self.W1(inputs)))  # (batch_size, time_steps, 1)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch_size, time_steps, 1)
        
        # Apply attention weights to the inputs
        context_vector = inputs * attention_weights
        
        return context_vector, attention_weights

def residual_block(x, filters, kernel_size=(3, 3), strides=(1, 1), use_bias=True):
    """Create a residual block for improved gradient flow."""
    shortcut = x
    
    # First convolution layer
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias, 
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Second convolution layer
    x = Conv2D(filters, kernel_size, padding='same', use_bias=use_bias, 
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    
    # If dimensions changed, adjust shortcut with 1x1 conv
    if shortcut.shape[-1] != filters or strides != (1, 1):
        shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=use_bias)(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add shortcut to result
    x = Add()([x, shortcut])
    x = LeakyReLU(alpha=0.1)(x)
    
    return x

def build_advanced_cnn_feature_extractor(input_shape):
    """
    Build an advanced CNN model for feature extraction from individual frames.
    Uses residual connections and more efficient architecture.
    
    Args:
        input_shape: Shape of the input images (height, width, channels)
        
    Returns:
        A CNN model that extracts features from images
    """
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv2D(32, (7, 7), strides=(2, 2), padding='same', kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # First stack of residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = SpatialDropout2D(0.1)(x)
    
    # Second stack of residual blocks
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = SpatialDropout2D(0.1)(x)
    
    # Third stack of residual blocks
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    
    # Global pooling to reduce parameters
    x = GlobalAveragePooling2D()(x)
    
    # Dense layer for feature extraction
    x = Dense(512, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.3)(x)
    
    model = Model(inputs=inputs, outputs=x)
    return model

def build_cnn_lstm_model(num_classes):
    """
    Build an improved CNN+LSTM model for sign language recognition.
    
    Args:
        num_classes: Number of classes (sentences) to recognize
        
    Returns:
        A compiled CNN+LSTM model
    """
    # Input shape for individual frames
    frame_input_shape = (*IMAGE_SIZE, 3)
    
    # Build the advanced CNN feature extractor
    cnn_model = build_advanced_cnn_feature_extractor(frame_input_shape)
    
    # Use the Functional API to build the full model
    input_shape = (MAX_SEQUENCE_LENGTH, *IMAGE_SIZE, 3)
    
    # Main input - sequence of frames
    inputs = Input(shape=input_shape, name='input_layer')
    
    # Apply the CNN to each frame in the sequence
    time_distributed_cnn = TimeDistributed(cnn_model)(inputs)
    
    # Use bidirectional LSTM for better sequence modeling
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, 
                          recurrent_dropout=0.2, 
                          dropout=0.2))(time_distributed_cnn)
    
    # Apply temporal attention to focus on important frames
    attention_layer = TemporalAttention(LSTM_UNITS)
    x, attention_weights = attention_layer(x)
    
    # Second bidirectional LSTM layer
    x = Bidirectional(LSTM(LSTM_UNITS, 
                          recurrent_dropout=0.2, 
                          dropout=0.2))(x)
    
    # Final classification layers
    x = Dense(256, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    x = Dense(128, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model with a lower learning rate for better stability
    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_callbacks(checkpoint_path):
    """
    Get improved training callbacks with better early stopping and LR scheduling.
    
    Args:
        checkpoint_path: Path to save the best model checkpoint
        
    Returns:
        List of callbacks for training
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    callbacks = [
        # Save the best model
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Reduce learning rate when plateau is reached - more patient and gradual reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # More gradual reduction
            patience=8,  # More patience
            min_lr=1e-6,
            verbose=1
        ),
        
        # Early stopping with more patience to allow the model to converge
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,  # More patience
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir='logs',
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    return callbacks

def create_model_summary(model, file_path):
    """
    Create a text file with the model summary.
    
    Args:
        model: The Keras model
        file_path: Path to save the summary
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Get the model summary
    from io import StringIO
    summary_string = StringIO()
    model.summary(print_fn=lambda x: summary_string.write(x + '\n'))
    
    # Write to file
    with open(file_path, 'w') as f:
        f.write(summary_string.getvalue())
    
    print(f"Model summary saved to {file_path}")

if __name__ == "__main__":
    # If run as a script, build and save the model summary
    model = build_cnn_lstm_model(num_classes=15)  # Example with 15 classes
    model.summary()
    create_model_summary(model, 'models/model_summary.txt') 