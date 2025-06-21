#!/usr/bin/env python3
"""
Test script to verify the project setup is working correctly.
"""

import os
import sys
import importlib

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")
    
    # Add src to path
    src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    sys.path.append(src_path)
    
    modules_to_test = [
        'config',
        'models.cnn_lstm_model',
        'data_utils.data_preprocessor',
        'training.train_model',
        'evaluation.evaluate_without_loading',
        'utils.webcam_inference',
        'utils.emotion_demo',
        'utils.test_webcam'
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    return failed_imports

def test_directories():
    """Test that all required directories exist."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'src',
        'src/models',
        'src/data_utils',
        'src/training',
        'src/evaluation',
        'src/utils',
        'models',
        'models/checkpoints',
        'models/finetuned',
        'data',
        'data/raw',
        'data/processed',
        'logs',
        'evaluation_results',
        'inference_results',
        'recorded_gestures'
    ]
    
    missing_dirs = []
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ {directory}")
        else:
            print(f"✗ {directory}")
            missing_dirs.append(directory)
    
    return missing_dirs

def test_files():
    """Test that all required files exist."""
    print("\nTesting required files...")
    
    required_files = [
        'requirements.txt',
        'README.md',
        'setup.py',
        'Makefile',
        '.gitignore',
        'run_pipeline.py',
        'src/__init__.py',
        'src/config.py',
        'src/models/__init__.py',
        'src/models/cnn_lstm_model.py',
        'src/data_utils/__init__.py',
        'src/data_utils/data_preprocessor.py',
        'src/training/__init__.py',
        'src/training/train_model.py',
        'src/evaluation/__init__.py',
        'src/evaluation/evaluate_without_loading.py',
        'src/utils/__init__.py',
        'src/utils/webcam_inference.py',
        'src/utils/emotion_demo.py',
        'src/utils/test_webcam.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")
            missing_files.append(file_path)
    
    return missing_files

def main():
    """Run all tests."""
    print("=" * 50)
    print("SIGN LANGUAGE RECOGNITION PROJECT SETUP TEST")
    print("=" * 50)
    
    # Test imports
    failed_imports = test_imports()
    
    # Test directories
    missing_dirs = test_directories()
    
    # Test files
    missing_files = test_files()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    if not failed_imports and not missing_dirs and not missing_files:
        print("🎉 All tests passed! The project is set up correctly.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Place your dataset in data/raw/")
        print("3. Update DATASET_ROOT in src/config.py if needed")
        print("4. Run the pipeline: python run_pipeline.py")
    else:
        print("❌ Some tests failed:")
        
        if failed_imports:
            print(f"  - Failed imports: {len(failed_imports)}")
            for imp in failed_imports:
                print(f"    * {imp}")
        
        if missing_dirs:
            print(f"  - Missing directories: {len(missing_dirs)}")
            for dir_path in missing_dirs:
                print(f"    * {dir_path}")
        
        if missing_files:
            print(f"  - Missing files: {len(missing_files)}")
            for file_path in missing_files:
                print(f"    * {file_path}")
        
        print("\nPlease fix the issues above before proceeding.")

if __name__ == "__main__":
    main() 