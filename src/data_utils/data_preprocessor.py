"""
Data preprocessing module for the Sign Language Recognition Project.
This module handles the preparation of the dataset for training.
"""

import os
import cv2
import numpy as np
import json
import shutil
import sys
import glob
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATASET_ROOT, PROCESSED_DATA_DIR, SELECTED_SENTENCES,
    MAX_SEQUENCE_LENGTH, IMAGE_SIZE, VALIDATION_SPLIT, TEST_SPLIT,
    LABEL_MAP_PATH
)

def create_label_map():
    """
    Create a mapping between sentence labels and their indices.
    """
    label_map = {sentence: idx for idx, sentence in enumerate(SELECTED_SENTENCES)}
    
    # Save the label map to file
    os.makedirs(os.path.dirname(LABEL_MAP_PATH), exist_ok=True)
    with open(LABEL_MAP_PATH, 'w') as f:
        json.dump(label_map, f, indent=4)
    
    print(f"Created label map with {len(label_map)} classes and saved to {LABEL_MAP_PATH}")
    return label_map

def load_and_preprocess_sequence(sequence_path):
    """
    Load and preprocess a sequence of images.
    
    Args:
        sequence_path: Path to the directory containing the sequence images
        
    Returns:
        A numpy array of preprocessed images with shape (MAX_SEQUENCE_LENGTH, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    """
    # Get all jpg or png files
    image_files = sorted(
        glob.glob(os.path.join(sequence_path, "*.jpg")) + 
        glob.glob(os.path.join(sequence_path, "*.png"))
    )
    
    # Skip if not enough images
    if len(image_files) < MAX_SEQUENCE_LENGTH:
        return None
    
    # Sample frames evenly if we have more than we need
    if len(image_files) > MAX_SEQUENCE_LENGTH:
        indices = np.linspace(0, len(image_files) - 1, MAX_SEQUENCE_LENGTH, dtype=int)
        image_files = [image_files[i] for i in indices]
    
    # Load and preprocess each image
    preprocessed_images = []
    for img_path in image_files:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Resize to target size
        img = cv2.resize(img, IMAGE_SIZE)
        
        # Convert to RGB (OpenCV uses BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        preprocessed_images.append(img)
    
    # Ensure we have exactly MAX_SEQUENCE_LENGTH frames
    if len(preprocessed_images) != MAX_SEQUENCE_LENGTH:
        return None
    
    return np.array(preprocessed_images)

def apply_augmentation(sequence, num_augmentations=1):
    """
    Apply multiple augmentation techniques to a sequence.
    
    Args:
        sequence: Original sequence of frames
        num_augmentations: Number of augmented sequences to generate
        
    Returns:
        List of augmented sequences
    """
    augmented_sequences = []
    
    for _ in range(num_augmentations):
        # Choose random augmentation techniques
        augmentation_types = random.sample([
            'flip', 'brightness', 'contrast', 'saturation', 
            'rotation', 'translation', 'zoom', 'noise'
        ], k=random.randint(1, 3))  # Apply 1-3 augmentations randomly
        
        aug_sequence = sequence.copy()
        
        # Apply each chosen augmentation
        for aug_type in augmentation_types:
            if aug_type == 'flip' and random.random() < 0.5:
                # Horizontal flip
                aug_sequence = np.array([np.fliplr(frame) for frame in aug_sequence])
                
            elif aug_type == 'brightness':
                # Random brightness adjustment
                factor = random.uniform(0.75, 1.25)
                aug_sequence = np.clip(aug_sequence * factor, 0.0, 1.0)
                
            elif aug_type == 'contrast':
                # Random contrast adjustment
                factor = random.uniform(0.75, 1.25)
                mean = np.mean(aug_sequence, axis=(1, 2), keepdims=True)
                aug_sequence = np.clip((aug_sequence - mean) * factor + mean, 0.0, 1.0)
                
            elif aug_type == 'saturation':
                # Random saturation adjustment (only affects color channels)
                factor = random.uniform(0.75, 1.25)
                for i in range(len(aug_sequence)):
                    hsv = cv2.cvtColor((aug_sequence[i] * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
                    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * factor, 0, 255).astype(np.uint8)
                    aug_sequence[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
                    
            elif aug_type == 'rotation':
                # Random rotation
                angle = random.uniform(-15, 15)
                for i in range(len(aug_sequence)):
                    h, w = aug_sequence[i].shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    aug_sequence[i] = cv2.warpAffine(aug_sequence[i], M, (w, h), borderMode=cv2.BORDER_REFLECT)
                    
            elif aug_type == 'translation':
                # Random translation
                tx, ty = random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)
                for i in range(len(aug_sequence)):
                    h, w = aug_sequence[i].shape[:2]
                    M = np.float32([[1, 0, w * tx], [0, 1, h * ty]])
                    aug_sequence[i] = cv2.warpAffine(aug_sequence[i], M, (w, h), borderMode=cv2.BORDER_REFLECT)
                    
            elif aug_type == 'zoom':
                # Random zoom
                factor = random.uniform(0.9, 1.1)
                for i in range(len(aug_sequence)):
                    h, w = aug_sequence[i].shape[:2]
                    crop_h, crop_w = int(h * factor), int(w * factor)
                    start_h = max(0, (h - crop_h) // 2)
                    start_w = max(0, (w - crop_w) // 2)
                    end_h = min(h, start_h + crop_h)
                    end_w = min(w, start_w + crop_w)
                    cropped = aug_sequence[i][start_h:end_h, start_w:end_w]
                    aug_sequence[i] = cv2.resize(cropped, (w, h))
                    
            elif aug_type == 'noise':
                # Add random noise
                noise_level = random.uniform(0.01, 0.05)
                noise = np.random.normal(0, noise_level, aug_sequence.shape)
                aug_sequence = np.clip(aug_sequence + noise, 0.0, 1.0)
        
        augmented_sequences.append(aug_sequence)
    
    return augmented_sequences

def generate_synthetic_sequences(sequences, labels, min_samples_per_class=50):
    """
    Generate synthetic sequences for underrepresented classes to ensure class balance.
    
    Args:
        sequences: List of original sequences
        labels: List of corresponding labels
        min_samples_per_class: Minimum number of samples per class
        
    Returns:
        Tuple of (augmented_sequences, augmented_labels)
    """
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    # Count samples per class
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique_labels, counts))
    
    # Add synthetic data for underrepresented classes
    synthetic_sequences = []
    synthetic_labels = []
    
    for label, count in class_counts.items():
        if count < min_samples_per_class:
            # Get sequences for this class
            class_sequences = sequences[labels == label]
            
            # How many additional samples we need
            samples_needed = min_samples_per_class - count
            
            # Generate more samples through augmentation
            for i in range(min(len(class_sequences), (samples_needed + len(class_sequences) - 1) // len(class_sequences))):
                # How many augmentations to create from this sequence
                augmentations_per_seq = min(samples_needed, 10)  # Cap at 10 per sequence to avoid overfitting
                samples_needed -= augmentations_per_seq
                
                # Create augmentations
                augmented = apply_augmentation(class_sequences[i % len(class_sequences)], augmentations_per_seq)
                synthetic_sequences.extend(augmented)
                synthetic_labels.extend([label] * len(augmented))
                
                if samples_needed <= 0:
                    break
    
    # Combine original and synthetic data
    combined_sequences = np.concatenate([sequences, np.array(synthetic_sequences)], axis=0)
    combined_labels = np.concatenate([labels, np.array(synthetic_labels)], axis=0)
    
    # Shuffle to mix original and synthetic
    combined_sequences, combined_labels = shuffle(combined_sequences, combined_labels, random_state=42)
    
    return combined_sequences, combined_labels

def prepare_dataset():
    """
    Prepare the dataset by:
    1. Selecting only the specified sentences
    2. Preprocessing all sequences
    3. Applying augmentation and balancing classes
    4. Splitting into train, validation, and test sets
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Create label map
    label_map = create_label_map()
    
    # Lists to store sequences and labels
    all_sequences = []
    all_labels = []
    
    # Keep track of samples per class
    class_counts = {idx: 0 for idx in range(len(SELECTED_SENTENCES))}
    
    # Process each selected sentence
    for sentence in tqdm(SELECTED_SENTENCES, desc="Processing sentences"):
        sentence_path = os.path.join(DATASET_ROOT, sentence)
        if not os.path.exists(sentence_path):
            print(f"Warning: Path not found for sentence '{sentence}'")
            continue
        
        # Get all person directories for this sentence
        person_dirs = [d for d in os.listdir(sentence_path) 
                      if os.path.isdir(os.path.join(sentence_path, d))]
        
        # Process each person's sequence
        for person in person_dirs:
            person_path = os.path.join(sentence_path, person)
            sequence = load_and_preprocess_sequence(person_path)
            
            if sequence is not None:
                class_idx = label_map[sentence]
                all_sequences.append(sequence)
                all_labels.append(class_idx)
                class_counts[class_idx] += 1
                
                # Add basic augmentation for each original sequence
                augmented_sequences = apply_augmentation(sequence, num_augmentations=4)
                all_sequences.extend(augmented_sequences)
                all_labels.extend([class_idx] * len(augmented_sequences))
                class_counts[class_idx] += len(augmented_sequences)
    
    # Balance the dataset
    print("Class counts before balancing:", class_counts)
    all_sequences, all_labels = generate_synthetic_sequences(
        all_sequences, all_labels, min_samples_per_class=50
    )
    
    # Convert to numpy arrays
    X = np.array(all_sequences)
    y = np.array(all_labels)
    
    # Check class distribution after balancing
    unique_labels, counts = np.unique(y, return_counts=True)
    balanced_class_counts = dict(zip(unique_labels, counts))
    print("Class counts after balancing:", balanced_class_counts)
    
    # Calculate min samples per class for stratification
    min_samples_per_class = min(counts)
    
    print(f"Dataset prepared. Total sequences: {len(all_sequences)}")
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
    print(f"Minimum samples per class: {min_samples_per_class}")
    
    # Ensure we have enough samples per class for stratified split
    adjusted_test_split = max(TEST_SPLIT, (2 * len(unique_labels)) / len(all_sequences))
    adjusted_test_split = min(adjusted_test_split, 0.2)  # Cap at 20% to avoid overfitting
    
    print(f"Using test split of {adjusted_test_split:.3f} to ensure each class has enough test samples")
    
    # First split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=adjusted_test_split, stratify=y, random_state=42
    )
    
    # Then split train+val into train and val
    adjusted_val_split = VALIDATION_SPLIT/(1-adjusted_test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, 
        test_size=adjusted_val_split,
        stratify=y_trainval, 
        random_state=42
    )
    
    # Convert labels to one-hot encoding
    y_train_onehot = np.eye(len(SELECTED_SENTENCES))[y_train]
    y_val_onehot = np.eye(len(SELECTED_SENTENCES))[y_val]
    y_test_onehot = np.eye(len(SELECTED_SENTENCES))[y_test]
    
    print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    return X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot, label_map

def save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Save the processed data to disk for faster loading in the future.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), y_test)
    
    print(f"Preprocessed data saved to {PROCESSED_DATA_DIR}")

def load_processed_data():
    """
    Load preprocessed data from disk if available.
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test) or None if not found
    """
    try:
        X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
        y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
        X_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'))
        y_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'))
        X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
        
        print(f"Loaded preprocessed data from {PROCESSED_DATA_DIR}")
        print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    except FileNotFoundError:
        print("Preprocessed data not found.")
        return None

def get_dataset():
    """
    Get the dataset - load preprocessed data if available, or prepare it if not.
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, label_map)
    """
    # First check if we need to regenerate the dataset
    force_regenerate = os.environ.get('FORCE_REGENERATE_DATA', '0') == '1'
    
    if not force_regenerate:
        data = load_processed_data()
        
        if data is not None:
            # Load label map
            with open(LABEL_MAP_PATH, 'r') as f:
                label_map = json.load(f)
            return *data, label_map
    
    # If regenerating or data not found, prepare it from scratch
    print("Preparing dataset from scratch...")
    X_train, y_train, X_val, y_val, X_test, y_test, label_map = prepare_dataset()
    save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, label_map

if __name__ == "__main__":
    # If run as a script, prepare and save the dataset
    X_train, y_train, X_val, y_val, X_test, y_test, label_map = get_dataset()
    print("Dataset preparation complete.") 