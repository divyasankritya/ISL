#!/usr/bin/env python3
"""
Integrated Emotion and Sign Language Recognition System.
This script combines both emotion detection and sign language recognition in real-time.
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
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

# Define the custom TemporalAttention layer needed for the sign language model
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

# Define emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def load_sign_language_model(model_path):
    """
    Load the fine-tuned sign language recognition model.
    
    Args:
        model_path: Path to the saved model file (.h5)
        
    Returns:
        The loaded model
    """
    if not os.path.exists(model_path):
        print(f"Error: Sign language model file not found: {model_path}")
        sys.exit(1)
        
    print(f"Loading sign language model from {model_path}...")
    
    # Load the model with custom objects to handle the TemporalAttention layer
    custom_objects = {'TemporalAttention': TemporalAttention}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print("Sign language model loaded successfully.")
    
    return model

def load_emotion_model(model_path):
    """
    Load the emotion detection model.
    
    Args:
        model_path: Path to the saved model file (.h5)
        
    Returns:
        The loaded model
    """
    if not os.path.exists(model_path):
        print(f"Error: Emotion model file not found: {model_path}")
        sys.exit(1)
        
    print(f"Loading emotion model from {model_path}...")
    
    # Load the model
    model = load_model(model_path)
    print("Emotion model loaded successfully.")
    
    return model

def preprocess_frame_for_sign_language(frame, target_size=IMAGE_SIZE):
    """
    Preprocess a frame for the sign language model.
    
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

def preprocess_face_for_emotion(face):
    """
    Preprocess a face image for emotion detection.
    
    Args:
        face: The detected face region from the frame
        
    Returns:
        Preprocessed face for emotion model
    """
    # Resize to 48x48 (common size for emotion models)
    face = cv2.resize(face, (48, 48))
    
    # Convert to grayscale if it's not already
    if len(face.shape) == 3 and face.shape[2] == 3:
        face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    
    # Normalize pixel values
    face = face.astype(np.float32) / 255.0
    
    # Reshape for model input: (batch_size, height, width, channels)
    face = face.reshape(1, 48, 48, 1)
    
    return face

def detect_emotion(emotion_model, face):
    """
    Detect emotion from a face image.
    
    Args:
        emotion_model: The loaded emotion model
        face: The detected face region from the frame
        
    Returns:
        Tuple of (emotion, confidence)
    """
    # Preprocess the face
    processed_face = preprocess_face_for_emotion(face)
    
    # Make prediction
    emotion_probs = emotion_model.predict(processed_face, verbose=0)[0]
    
    # Get the top emotion
    emotion_idx = np.argmax(emotion_probs)
    emotion = EMOTIONS[emotion_idx]
    confidence = emotion_probs[emotion_idx]
    
    # Get top 3 emotions
    top_indices = np.argsort(emotion_probs)[::-1][:3]
    top_emotions = [(EMOTIONS[idx], emotion_probs[idx]) for idx in top_indices]
    
    return emotion, confidence, top_emotions

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

def get_sign_language_label_map():
    """
    Get the mapping of class indices to labels for sign language.
    
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

def run_integrated_recognition(sign_language_model, emotion_model, sl_label_map, 
                            confidence_threshold=0.7, camera_index=1):
    """
    Run real-time inference for both emotion and sign language recognition.
    
    Args:
        sign_language_model: The loaded sign language model
        emotion_model: The loaded emotion model
        sl_label_map: Mapping from class indices to labels for sign language
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
    
    # Initialize face cascade classifier for face detection
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("Face detection initialized.")
    except:
        face_cascade = None
        print("Warning: Could not initialize face detection. Emotion recognition may not work properly.")
    
    # Initialize frame buffer for sign language recognition
    frame_buffer = deque(maxlen=MAX_SEQUENCE_LENGTH)
    
    # Fill buffer with initial (blank) frames
    for _ in range(MAX_SEQUENCE_LENGTH):
        blank_frame = np.zeros((*IMAGE_SIZE, 3), dtype=np.float32)
        frame_buffer.append(blank_frame)
    
    # Variables for controlling prediction frequency
    last_sl_prediction_time = time.time()
    last_emotion_prediction_time = time.time()
    sl_prediction_interval = 1.0  # Seconds between sign language predictions
    emotion_prediction_interval = 0.5  # Seconds between emotion predictions
    current_sl_prediction = None
    current_sl_confidence = 0.0
    current_emotion = None
    current_emotion_confidence = 0.0
    
    # Variables for recording mode
    recording = False
    recording_start_time = 0
    recording_frames = []
    recording_length = 3  # Seconds to record
    
    print("\nControls:")
    print("- Press 'q' to quit")
    print("- Press 'r' to start/stop recording a specific sequence")
    print(f"- Sign language predictions will update every {sl_prediction_interval} seconds")
    print(f"- Emotion predictions will update every {emotion_prediction_interval} seconds")
    print(f"- Only predictions with confidence above {confidence_threshold*100}% will be shown")
    print("\nStarting real-time recognition. Please perform gestures in front of the camera.")
    
    while True:
        # Read a frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break
        
        # Flip frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        
        # Copy the original frame for display
        display_frame = frame.copy()
        
        # Preprocess the frame for sign language recognition
        processed_frame = preprocess_frame_for_sign_language(frame)
        
        # Add to sign language frame buffer
        frame_buffer.append(processed_frame)
        
        # Detect faces for emotion recognition
        current_time = time.time()
        if face_cascade and current_time - last_emotion_prediction_time >= emotion_prediction_interval:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # If faces are found, process the largest one
            if len(faces) > 0:
                # Find the largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Detect emotion
                emotion, emotion_confidence, top_emotions = detect_emotion(emotion_model, face_roi)
                
                # Update current emotion if confidence is high enough
                if emotion_confidence >= confidence_threshold:
                    current_emotion = emotion
                    current_emotion_confidence = emotion_confidence
                
                # Draw rectangle around the face
                color = (0, 255, 0)  # Green
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                # Display emotion above the face
                if current_emotion:
                    emotion_text = f"{current_emotion}: {current_emotion_confidence*100:.1f}%"
                    cv2.putText(display_frame, emotion_text, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            last_emotion_prediction_time = current_time
        
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
                
                # Save as individual frames
                for i, rec_frame in enumerate(recording_frames):
                    frame_path = os.path.join(save_path, f"gesture_{timestamp}_{i:03d}.jpg")
                    cv2.imwrite(frame_path, rec_frame)
                
                print(f"Frames saved to {save_path}")
                recording_frames = []
        
        # Run sign language prediction at fixed intervals
        if current_time - last_sl_prediction_time >= sl_prediction_interval:
            # Convert buffer to numpy array
            sequence = np.array(list(frame_buffer))
            
            # Run prediction with TTA
            pred_probs = test_time_augmentation(sign_language_model, sequence)
            
            # Get predicted class and confidence
            pred_idx = np.argmax(pred_probs)
            confidence = pred_probs[pred_idx]
            
            # Update prediction if confidence is high enough
            if confidence >= confidence_threshold:
                current_sl_prediction = sl_label_map.get(pred_idx, f"Class {pred_idx}")
                current_sl_confidence = confidence
            
            # Get top 3 predictions for display
            top_indices = np.argsort(pred_probs)[::-1][:3]
            top_sl_predictions = [(sl_label_map.get(idx, f"Class {idx}"), pred_probs[idx]) for idx in top_indices]
            
            last_sl_prediction_time = current_time
        
        # Create prediction display area
        prediction_height = 160
        prediction_box = np.zeros((prediction_height, frame_width, 3), dtype=np.uint8)
        
        # Draw dividing line between emotion and sign language sections
        cv2.line(prediction_box, (0, prediction_height//2), (frame_width, prediction_height//2), 
                (50, 50, 50), 2)
        
        # Draw title for each section
        cv2.putText(prediction_box, "Emotion Recognition", (20, 25), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(prediction_box, "Sign Language Recognition", (20, prediction_height//2 + 25), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Display current emotion if available
        if current_emotion and current_emotion_confidence >= confidence_threshold:
            cv2.putText(prediction_box, f"Emotion: {current_emotion}", 
                      (220, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(prediction_box, f"Confidence: {current_emotion_confidence*100:.1f}%",
                      (220, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(prediction_box, "No emotion detected", 
                      (220, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
            
        # Display current sign language prediction if available
        if current_sl_prediction and current_sl_confidence >= confidence_threshold:
            cv2.putText(prediction_box, f"Sign: {current_sl_prediction}",
                      (220, prediction_height//2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(prediction_box, f"Confidence: {current_sl_confidence*100:.1f}%",
                      (220, prediction_height//2 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(prediction_box, "Waiting for gesture...",
                      (220, prediction_height//2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        
        # Draw recording indicator if recording
        if recording:
            elapsed = time.time() - recording_start_time
            remaining = max(0, recording_length - elapsed)
            cv2.putText(prediction_box, f"RECORDING: {remaining:.1f}s",
                      (frame_width - 220, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.7, (0, 0, 255), 2)
        
        # Combine frame with prediction box
        combined_frame = np.vstack((display_frame, prediction_box))
        
        # Display the result
        cv2.imshow("Integrated Emotion and Sign Language Recognition", combined_frame)
        
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
                
                # Save the recorded frames
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
    print("Recognition stopped.")

def main():
    parser = argparse.ArgumentParser(description="Integrated Emotion and Sign Language Recognition System")
    
    # Model paths
    parser.add_argument('--sl-model', type=str, 
                      default='/Users/nilopakumar/Desktop/ISL_NEW/sign_language_project/models/finetuned/final_model.h5',
                      help='Path to the sign language model')
    
    parser.add_argument('--emotion-model', type=str, 
                      default='/Users/nilopakumar/Desktop/ISL_NEW/models/emotion_models.h5',
                      help='Path to the emotion recognition model')
    
    # Confidence threshold
    parser.add_argument('--threshold', type=float, default=0.7,
                      help='Confidence threshold for displaying predictions (0.0-1.0)')
    
    # Camera index
    parser.add_argument('--camera', type=int, default=1,
                      help='Camera index to use (default: 1, try 0 if not working)')
    
    args = parser.parse_args()
    
    # Load the sign language model
    sign_language_model = load_sign_language_model(args.sl_model)
    
    # Load the emotion model
    emotion_model = load_emotion_model(args.emotion_model)
    
    # Get label map for sign language
    sl_label_map = get_sign_language_label_map()
    
    # Run integrated recognition
    run_integrated_recognition(
        sign_language_model,
        emotion_model,
        sl_label_map,
        args.threshold,
        args.camera
    )

if __name__ == "__main__":
    main() 