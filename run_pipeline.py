#!/usr/bin/env python3
"""
Complete pipeline script for the Sign Language Recognition project.
This script runs all steps of the pipeline: data preprocessing, training, and evaluation.
"""

import os
import sys
import time
import argparse
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from config import (
    PROCESSED_DATA_DIR, CHECKPOINT_PATH, 
    FINAL_MODEL_PATH, HISTORY_PATH
)

def run_preprocessing():
    """Run the data preprocessing step"""
    print("\n===== STEP 1: DATA PREPROCESSING =====\n")
    from data_utils.data_preprocessor import get_dataset
    
    start_time = time.time()
    X_train, y_train, X_val, y_val, X_test, y_test, label_map = get_dataset()
    
    preprocessing_time = time.time() - start_time
    print(f"\nPreprocessing completed in {preprocessing_time:.2f} seconds")
    print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    return preprocessing_time

def run_training():
    """Run the model training step"""
    print("\n===== STEP 2: MODEL TRAINING =====\n")
    from training.train_model import train_model, plot_training_history
    
    start_time = time.time()
    model, history = train_model()
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Plot training history
    plot_training_history(history)
    
    return model, training_time

def run_evaluation():
    """Run model evaluation."""
    print("\n===== STEP 3: MODEL EVALUATION =====\n")
    
    # Import at runtime to avoid circular imports
    from evaluation.evaluate_without_loading import run_evaluation as evaluate
    
    try:
        test_acc = evaluate()
        print(f"\nEvaluation complete with test accuracy: {test_acc:.4f}")
        return test_acc
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nEvaluation failed. Please check the error message above.")
        return None

def create_report(preprocessing_time, training_time, evaluation_time):
    """Create a summary report of the pipeline run."""
    print("\n===== PIPELINE SUMMARY =====\n")
    
    # Convert None values to 0 for calculation
    p_time = preprocessing_time if preprocessing_time is not None else 0
    t_time = training_time if training_time is not None else 0
    e_time = evaluation_time if evaluation_time is not None else 0
    
    total_time = p_time + t_time + e_time
    
    print(f"Preprocessing time: {p_time:.2f} seconds" if preprocessing_time is not None else "Preprocessing: SKIPPED")
    print(f"Training time: {t_time:.2f} seconds" if training_time is not None else "Training: SKIPPED")
    print(f"Evaluation time: {e_time:.2f} seconds" if evaluation_time is not None else "Evaluation: FAILED")
    print(f"Total pipeline time: {total_time:.2f} seconds")
    
    # Check if model exists
    if os.path.exists(FINAL_MODEL_PATH):
        print(f"Model saved at: {FINAL_MODEL_PATH}")
    else:
        print("Warning: No model file found at the expected location.")
    
    # Check if evaluation results exist
    eval_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation_results')
    if os.path.exists(eval_results_dir) and len(os.listdir(eval_results_dir)) > 0:
        print(f"Evaluation results saved at: {eval_results_dir}")
    else:
        print("Warning: No evaluation results found.")
    
    print("\n===== PIPELINE COMPLETE =====\n")

def run_pipeline(skip_preprocessing=False, skip_training=False, skip_evaluation=False):
    """
    Run the complete pipeline.
    
    Args:
        skip_preprocessing: Whether to skip the preprocessing step
        skip_training: Whether to skip the training step
        skip_evaluation: Whether to skip the evaluation step
    """
    # Create a timestamp for this run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting pipeline run at {run_timestamp}")
    
    preprocessing_time = 0
    training_time = 0
    evaluation_time = 0
    
    # Step 1: Preprocessing
    if not skip_preprocessing:
        preprocessing_time = run_preprocessing()
    else:
        print("\n===== STEP 1: DATA PREPROCESSING [SKIPPED] =====\n")
    
    # Step 2: Training
    if not skip_training:
        _, training_time = run_training()
    else:
        print("\n===== STEP 2: MODEL TRAINING [SKIPPED] =====\n")
    
    # Step 3: Evaluation
    if not skip_evaluation:
        evaluation_time = run_evaluation()
    else:
        print("\n===== STEP 3: MODEL EVALUATION [SKIPPED] =====\n")
    
    # Create report
    create_report(preprocessing_time, training_time, evaluation_time)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the sign language recognition pipeline")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip the preprocessing step")
    parser.add_argument("--skip-training", action="store_true", help="Skip the training step")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip the evaluation step")
    
    args = parser.parse_args()
    
    # Set up GPU memory growth
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s), enabled memory growth")
        except RuntimeError as e:
            print(f"GPU error: {e}")
    
    # Run the pipeline
    run_pipeline(
        skip_preprocessing=args.skip_preprocessing,
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation
    )

if __name__ == "__main__":
    main() 