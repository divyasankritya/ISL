# Multimodal Sign Language and Emotion Recognition System

A deep learning system for real-time sign language recognition using a CNN-LSTM architecture with temporal attention. Includes optional emotion detection.

## Features

- Real-time sign language recognition (sentences)
- Optional facial emotion detection
- CNN-LSTM with temporal attention
- Data augmentation and test-time augmentation
- Webcam integration for live inference
- Recording capabilities for gesture analysis

## Project Structure

```
src/
  models/                # Model architectures
  data_utils/            # Data processing
  training/              # Training scripts
  evaluation/            # Evaluation scripts
  utils/                 # Utilities (inference, demo, etc.)
  config.py              # Configuration
models/                  # Saved models
data/                    # Raw and processed data
logs/                    # Training logs
evaluation_results/      # Evaluation outputs
inference_results/       # Inference outputs
recorded_gestures/       # Recorded gesture sequences
run_pipeline.py          # Main pipeline script
requirements.txt         # Dependencies
README.md                # This file
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/divyasankritya/Multimodal-Sign-Language-and-Emotion-Recognition-System.git
   cd Multimodal-Sign-Language-and-Emotion-Recognition-System
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip3 install -r requirements.txt
   ```

4. **Prepare your dataset**
   - Place the ISL_CSLRT_Corpus dataset in `data/raw/`
   - Update `DATASET_ROOT` in `src/config.py` if needed

## Usage

**Full pipeline (preprocessing, training, evaluation):**
```bash
python3 run_pipeline.py
```

**Skip steps if needed:**
```bash
python3 run_pipeline.py --skip-preprocessing
python3 run_pipeline.py --skip-preprocessing --skip-training
```

**Webcam inference (Sign Language Recognition):**
```bash
python3 src/utils/webcam_inference.py --model models/finetuned/final_model.h5
```

**Emotion detection:**
```bash
python3 src/utils/emotion_demo.py --webcam
```

**Integrated recognition (Sign Language + Emotion):**
```bash
python3 src/utils/integrated_recognition.py --sl-model models/finetuned/final_model.h5 --emotion-model models/emotion_models.h5
```

**Test webcam:**
```bash
python3 src/utils/test_webcam.py
```

## Configuration

Edit `src/config.py` to change:
- Dataset path (`DATASET_ROOT`)
- Model/training parameters (`IMAGE_SIZE`, `BATCH_SIZE`, `EPOCHS`, etc.)

## Troubleshooting

- **Camera not working:** Run `python3 src/utils/test_webcam.py`
- **CUDA/GPU issues:** Ensure correct TensorFlow version and CUDA drivers
- **Memory issues:** Reduce `BATCH_SIZE` or `IMAGE_SIZE` in config
- **Dataset issues:** Check `DATASET_ROOT` and folder structure
