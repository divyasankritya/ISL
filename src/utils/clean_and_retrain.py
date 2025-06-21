#!/usr/bin/env python3
"""
Script to clean existing processed data and retrain the model with improved parameters.
"""

import os
import sys
import shutil
import argparse
import time
from datetime import datetime

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import PROCESSED_DATA_DIR, MODEL_SAVE_DIR
from run_pipeline import run_pipeline

def clean_processed_data():
    """
    Remove existing processed data to force regeneration.
    """
    if os.path.exists(PROCESSED_DATA_DIR):
        print(f"Removing existing processed data directory: {PROCESSED_DATA_DIR}")
        shutil.rmtree(PROCESSED_DATA_DIR)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        print("Processed data directory cleaned.")
    else:
        print("No existing processed data found.")

def backup_models():
    """
    Backup existing models before retraining.
    """
    if os.path.exists(MODEL_SAVE_DIR):
        # Create backup directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{MODEL_SAVE_DIR}_backup_{timestamp}"
        
        print(f"Backing up existing models to: {backup_dir}")
        shutil.copytree(MODEL_SAVE_DIR, backup_dir)
        print("Model backup completed.")
    else:
        print("No existing models found to backup.")

def main():
    """
    Main function to clean data and retrain the model.
    """
    parser = argparse.ArgumentParser(description="Clean processed data and retrain the model")
    parser.add_argument("--skip-backup", action="store_true", help="Skip backing up existing models")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip the evaluation step")
    args = parser.parse_args()
    
    print("=" * 80)
    print("CLEANING AND RETRAINING ISL RECOGNITION MODEL")
    print("=" * 80)
    
    # Step 1: Backup existing models
    if not args.skip_backup:
        backup_models()
    else:
        print("Skipping model backup as requested.")
    
    # Step 2: Clean processed data
    clean_processed_data()
    
    # Step 3: Set environment variable to force data regeneration
    os.environ['FORCE_REGENERATE_DATA'] = '1'
    
    # Step 4: Run the pipeline
    print("\nStarting pipeline with regenerated data...\n")
    run_pipeline(
        skip_preprocessing=False,
        skip_training=False,
        skip_evaluation=args.skip_evaluation
    )
    
    print("\n" + "=" * 80)
    print("CLEANING AND RETRAINING COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main() 