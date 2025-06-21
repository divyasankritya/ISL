.PHONY: help install setup clean train evaluate inference webcam emotion test format lint

# Default target
help:
	@echo "Available commands:"
	@echo "  setup     - Set up the project environment and directories"
	@echo "  install   - Install dependencies"
	@echo "  clean     - Clean up generated files"
	@echo "  train     - Train the model"
	@echo "  evaluate  - Evaluate the model"
	@echo "  inference - Run batch inference"
	@echo "  webcam    - Run real-time webcam inference"
	@echo "  emotion   - Run emotion detection demo"
	@echo "  test      - Test webcam functionality"
	@echo "  format    - Format code with black"
	@echo "  lint      - Lint code with flake8"

# Setup project
setup:
	@echo "Setting up project directories..."
	mkdir -p data/{raw,processed}
	mkdir -p models/{checkpoints,finetuned}
	mkdir -p logs evaluation_results inference_results recorded_gestures
	@echo "Project setup complete!"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed!"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf logs/* evaluation_results/* inference_results/* recorded_gestures/*
	@echo "Cleanup complete!"

# Train model
train:
	@echo "Starting model training..."
	python run_pipeline.py

# Evaluate model
evaluate:
	@echo "Evaluating model..."
	python run_pipeline.py --skip-preprocessing --skip-training

# Run batch inference
inference:
	@echo "Running batch inference..."
	python src/utils/run_inference.py --model models/finetuned/final_model.h5 --evaluate

# Run webcam inference
webcam:
	@echo "Starting webcam inference..."
	python src/utils/webcam_inference.py --model models/finetuned/final_model.h5

# Run emotion detection
emotion:
	@echo "Starting emotion detection..."
	python src/utils/emotion_demo.py --webcam

# Test webcam
test:
	@echo "Testing webcam..."
	python src/utils/test_webcam.py

# Format code
format:
	@echo "Formatting code..."
	black src/ run_pipeline.py
	@echo "Code formatting complete!"

# Lint code
lint:
	@echo "Linting code..."
	flake8 src/ run_pipeline.py
	@echo "Linting complete!"

# Install development dependencies
install-dev:
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	pip install pytest black flake8 jupyter
	@echo "Development dependencies installed!"

# Run tests
test-all:
	@echo "Running tests..."
	python -m pytest tests/ -v

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	python -m venv venv
	@echo "Virtual environment created! Activate with: source venv/bin/activate"

# Full setup (venv + install + setup)
full-setup: venv
	@echo "Activating virtual environment and installing dependencies..."
	. venv/bin/activate && pip install -r requirements.txt
	@echo "Full setup complete!" 