#!/usr/bin/env python3
"""
Real-time webcam inference for the fine-tuned Sign Language Recognition Model.
This script captures webcam video, processes the frames, and runs inference in real-time.
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from collections import deque
import time
import argparse

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

# Import configuration
from config import IMAGE_SIZE, MAX_SEQUENCE_LENGTH
from data_utils.data_preprocessor import get_dataset

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

def preprocess_frame(frame, target_size=IMAGE_SIZE):
    """
    Preprocess a frame for the model.
    
    Args:
        frame: The input frame from webcam
        target_size: Target size for resizing
        
    Returns:
        Preprocessed frame
    """
    # Resize to the target size
    frame = cv2.resize(frame, target_size)
    
    # Convert to RGB if it's BGR
    if frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    # Normalize pixel values to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    
    return frame

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

def get_label_map():
    """
    Get the mapping of class indices to labels.
    
    Returns:
        Label map dictionary
    """
    # Get the label map from the dataset
    _, _, _, _, _, _, label_map = get_dataset()
    
    # Process label_map to ensure it's in the right format
    if isinstance(label_map, dict):
        if all(k.isdigit() for k in label_map.keys()):
            return {int(k): v for k, v in label_map.items()}
        else:
            return {v: k for k, v in label_map.items()}
    else:
        return {i: label for i, label in enumerate(label_map)}

def run_webcam_inference(model, label_map, confidence_threshold=0.7, camera_index=1):
    """
    Run real-time inference using webcam input.
    
    Args:
        model: The loaded model
        label_map: Mapping from class indices to labels
        confidence_threshold: Minimum confidence to display a prediction
        camera_index: Index of the camera to use
    """
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {camera_index}.")
        print("Trying alternative camera index...")
        alt_index = 0 if camera_index != 0 else 1
        print(f"Trying camera index {alt_index}...")
        cap = cv2.VideoCapture(alt_index)
        if not cap.isOpened():
            print("Error: Could not open any webcam.")
            print("Please check camera permissions and try again.")
            sys.exit(1)
    
    # Check if the camera is working and we can read frames
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Could not read from webcam.")
        print("Please check if another application is using the camera.")
        sys.exit(1)
    
    print("Webcam initialized successfully!")
    
    # Calculate frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Initialize frame buffer
    frame_buffer = deque(maxlen=MAX_SEQUENCE_LENGTH)
    
    # Fill buffer with initial (blank) frames
    for _ in range(MAX_SEQUENCE_LENGTH):
        blank_frame = np.zeros((*IMAGE_SIZE, 3), dtype=np.float32)
        frame_buffer.append(blank_frame)
    
    # Variables for controlling prediction frequency
    last_prediction_time = time.time()
    prediction_interval = 1.0  # Seconds between predictions
    current_prediction = None
    current_confidence = 0.0
    
    # Variables for recording mode
    recording = False
    recording_start_time = 0
    recording_frames = []
    recording_length = 3  # Seconds to record
    
    print("\nControls:")
    print("- Press 'q' to quit")
    print("- Press 'r' to start/stop recording a specific gesture sequence")
    print(f"- Predictions will update every {prediction_interval} seconds")
    print(f"- Only predictions with confidence above {confidence_threshold*100}% will be shown")
    print("\nStarting real-time inference. Please perform gestures in front of the camera.")
    
    while True:
        # Read a frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break
        
        # Flip frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        
        # Preprocess the frame
        processed_frame = preprocess_frame(frame)
        
        # Add to buffer
        frame_buffer.append(processed_frame)
        
        # If recording, add frame to recording buffer
        if recording:
            recording_frames.append(frame.copy())
            
            # Check if recording duration exceeded
            if time.time() - recording_start_time >= recording_length:
                recording = False
                print(f"Recording stopped. Saved {len(recording_frames)} frames.")
                
                # Save the recorded frames (optional)
                save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recorded_gestures')
                os.makedirs(save_path, exist_ok=True)
                timestamp = int(time.time())
                
                # Save as individual frames or create a video
                for i, rec_frame in enumerate(recording_frames):
                    frame_path = os.path.join(save_path, f"gesture_{timestamp}_{i:03d}.jpg")
                    cv2.imwrite(frame_path, rec_frame)
                
                print(f"Frames saved to {save_path}")
                recording_frames = []
        
        # Run prediction at fixed intervals
        current_time = time.time()
        if current_time - last_prediction_time >= prediction_interval:
            # Convert buffer to numpy array
            sequence = np.array(list(frame_buffer))
            
            # Run prediction with TTA
            pred_probs = test_time_augmentation(model, sequence)
            
            # Get predicted class and confidence
            pred_idx = np.argmax(pred_probs)
            confidence = pred_probs[pred_idx]
            
            # Update prediction if confidence is high enough
            if confidence >= confidence_threshold:
                current_prediction = label_map.get(pred_idx, f"Class {pred_idx}")
                current_confidence = confidence
            
            # Get top 3 predictions for display
            top_indices = np.argsort(pred_probs)[::-1][:3]
            top_predictions = [(label_map.get(idx, f"Class {idx}"), pred_probs[idx]) for idx in top_indices]
            
            last_prediction_time = current_time
        
        # Draw prediction box
        prediction_height = 120
        prediction_box = np.zeros((prediction_height, frame_width, 3), dtype=np.uint8)
        
        # Display current prediction if available
        if current_prediction and current_confidence >= confidence_threshold:
            # Draw main prediction
            cv2.putText(prediction_box, f"Prediction: {current_prediction}",
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(prediction_box, f"Confidence: {current_confidence*100:.1f}%",
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw recording indicator
            if recording:
                elapsed = time.time() - recording_start_time
                remaining = max(0, recording_length - elapsed)
                cv2.putText(prediction_box, f"RECORDING: {remaining:.1f}s",
                           (frame_width - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)
            
            # Draw top 3 predictions
            y_offset = 90
            for i, (pred, conf) in enumerate(top_predictions, 1):
                cv2.putText(prediction_box, f"{i}. {pred}: {conf*100:.1f}%",
                           (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                y_offset += 20
        else:
            # No high-confidence prediction
            cv2.putText(prediction_box, "Waiting for gesture...",
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw recording indicator
            if recording:
                elapsed = time.time() - recording_start_time
                remaining = max(0, recording_length - elapsed)
                cv2.putText(prediction_box, f"RECORDING: {remaining:.1f}s",
                           (frame_width - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)
        
        # Combine frame with prediction box
        display_frame = np.vstack((frame, prediction_box))
        
        # Display the result
        cv2.imshow("Sign Language Recognition", display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Quit if 'q' is pressed
        if key == ord('q'):
            break
            
        # Start/stop recording if 'r' is pressed
        elif key == ord('r'):
            if not recording:
                recording = True
                recording_start_time = time.time()
                recording_frames = [frame.copy()]
                print("Recording started. Perform your gesture.")
            else:
                recording = False
                print(f"Recording stopped early. Saved {len(recording_frames)} frames.")
                
                # Save the recorded frames (optional)
                save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recorded_gestures')
                os.makedirs(save_path, exist_ok=True)
                timestamp = int(time.time())
                
                # Save as individual frames
                for i, rec_frame in enumerate(recording_frames):
                    frame_path = os.path.join(save_path, f"gesture_{timestamp}_{i:03d}.jpg")
                    cv2.imwrite(frame_path, rec_frame)
                
                print(f"Frames saved to {save_path}")
                recording_frames = []
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam inference stopped.")

def main():
    parser = argparse.ArgumentParser(description="Real-time Sign Language Recognition using webcam")
    
    # Model path argument
    parser.add_argument('--model', type=str, 
                      default='/Users/nilopakumar/Desktop/ISL_NEW/sign_language_project/models/finetuned/final_model.h5',
                      help='Path to the fine-tuned model')
    
    # Confidence threshold
    parser.add_argument('--threshold', type=float, default=0.7,
                      help='Confidence threshold for displaying predictions (0.0-1.0)')
    
    # Camera index
    parser.add_argument('--camera', type=int, default=1,
                      help='Camera index to use (default: 1, try 0 if not working)')
    
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model)
    
    # Get label map
    label_map = get_label_map()
    
    # Run webcam inference
    run_webcam_inference(model, label_map, args.threshold, args.camera)

if __name__ == "__main__":
    main() 