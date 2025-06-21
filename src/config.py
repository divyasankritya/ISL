"""
Configuration file for the Sign Language Recognition Project.
Contains all the necessary parameters and paths.
"""

import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths - Update these paths according to your dataset location
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'data', 'raw', 'ISL_CSLRT_Corpus', 'Frames_Sentence_Level')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Selected 15 sentences to use
SELECTED_SENTENCES = [
    'he she is my friend',
    'what are you doing',
    'what happened',
    'my name is xxxxxxxx',
    'i like you i love you',
    'do not take it to the heart',
    'i am feeling bored',
    'can i help you',
    'serve the food',
    'what do you think',
    'do not make me angry',
    'could you please talk slower',
    'i am really grateful',
    'thank you so much',
    'hi how are you'
]

# Data processing parameters
MAX_SEQUENCE_LENGTH = 30  # Number of frames to use per gesture sequence
IMAGE_SIZE = (96, 96)     # Increased image size for better feature extraction
BATCH_SIZE = 16          # Smaller batch size to accommodate larger model
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Model parameters
CNN_FILTERS = [64, 128, 256, 512]  # More filters for better feature extraction
CNN_KERNEL_SIZE = (3, 3)
CNN_POOL_SIZE = (2, 2)
LSTM_UNITS = 256         # Increased LSTM units
DROPOUT_RATE = 0.4       # Slightly reduced dropout for better convergence
LEARNING_RATE = 0.0005   # Adjusted learning rate

# Training parameters
EPOCHS = 150             # More epochs with early stopping
EARLY_STOPPING_PATIENCE = 20
REDUCE_LR_PATIENCE = 8
REDUCE_LR_FACTOR = 0.5

# Data augmentation parameters
AUGMENTATION_FACTOR = 4  # Generate 4 augmented samples per original
MIN_SAMPLES_PER_CLASS = 50  # Ensure at least this many samples per class

# Advanced training parameters
USE_MIXED_PRECISION = True
USE_GRADIENT_ACCUMULATION = True
GRADIENT_ACCUMULATION_STEPS = 4
USE_MIXUP = True
MIXUP_ALPHA = 0.2
USE_TEST_TIME_AUGMENTATION = True
TTA_SAMPLES = 5

# Paths for saving models and results
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'models')
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, 'checkpoints', 'best_model.h5')
FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'finetuned', 'final_model.h5')
HISTORY_PATH = os.path.join(MODEL_SAVE_DIR, 'training_history.json')
LABEL_MAP_PATH = os.path.join(MODEL_SAVE_DIR, 'label_map.json')

# Logs and results directories
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
EVALUATION_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'evaluation_results')
INFERENCE_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'inference_results')
RECORDED_GESTURES_DIR = os.path.join(PROJECT_ROOT, 'recorded_gestures')

# Create directories if they don't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(EVALUATION_RESULTS_DIR, exist_ok=True)
os.makedirs(INFERENCE_RESULTS_DIR, exist_ok=True)
os.makedirs(RECORDED_GESTURES_DIR, exist_ok=True)

# Real-time inference parameters
CAMERA_ID = 0
COUNTDOWN_TIME = 3  # seconds
RECORDING_TIME = 3  # seconds to record gesture
FPS = 10            # Target FPS for recording 