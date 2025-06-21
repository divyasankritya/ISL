# Sign Language Recognition System

A comprehensive deep learning system for real-time sign language recognition using CNN-LSTM architecture with temporal attention mechanisms. This project also includes integrated emotion detection capabilities.

## 🚀 Features

- **Real-time Sign Language Recognition**: Process video streams and recognize 15 different sign language sentences
- **Emotion Detection**: Integrated facial emotion recognition system
- **Advanced Architecture**: CNN-LSTM model with temporal attention for better sequence understanding
- **Data Augmentation**: Comprehensive augmentation techniques for improved model robustness
- **Test-Time Augmentation**: Enhanced inference accuracy through multiple prediction averaging
- **Webcam Integration**: Real-time inference using webcam input
- **Recording Capabilities**: Save gesture sequences for analysis
- **Comprehensive Evaluation**: Detailed metrics and confusion matrix generation

## 📁 Project Structure

```
├── src/                          # Source code
│   ├── models/                   # Model architectures
│   │   ├── __init__.py
│   │   └── cnn_lstm_model.py     # CNN-LSTM model with temporal attention
│   ├── data_utils/               # Data processing utilities
│   │   ├── __init__.py
│   │   └── data_preprocessor.py  # Dataset preprocessing and augmentation
│   ├── training/                 # Training scripts
│   │   ├── __init__.py
│   │   └── train_model.py        # Model training pipeline
│   ├── evaluation/               # Evaluation scripts
│   │   ├── __init__.py
│   │   └── evaluate_without_loading.py  # Model evaluation
│   ├── utils/                    # Utility scripts
│   │   ├── __init__.py
│   │   ├── webcam_inference.py   # Real-time webcam inference
│   │   ├── integrated_recognition.py  # Combined SL + emotion recognition
│   │   ├── emotion_demo.py       # Emotion detection demo
│   │   ├── test_webcam.py        # Webcam testing utility
│   │   ├── run_inference.py      # Batch inference script
│   │   ├── partial_finetune.py   # Model fine-tuning
│   │   └── clean_and_retrain.py  # Clean training pipeline
│   └── config.py                 # Configuration parameters
├── models/                       # Saved models
│   ├── checkpoints/              # Training checkpoints
│   └── finetuned/                # Fine-tuned models
├── data/                         # Data directories
│   ├── raw/                      # Raw dataset
│   └── processed/                # Processed data
├── logs/                         # Training logs
├── evaluation_results/           # Evaluation outputs
├── inference_results/            # Inference outputs
├── recorded_gestures/            # Recorded gesture sequences
├── run_pipeline.py               # Main pipeline script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Webcam (for real-time inference)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sign-language-recognition
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your dataset**
   - Place your ISL_CSLRT_Corpus dataset in `data/raw/`
   - Update the `DATASET_ROOT` path in `src/config.py` if needed

## 🎯 Usage

### 1. Data Preprocessing

```bash
python -c "from src.data_utils.data_preprocessor import get_dataset; get_dataset()"
```

### 2. Model Training

```bash
python run_pipeline.py
```

Or run individual components:

```bash
# Training only
python run_pipeline.py --skip-preprocessing --skip-evaluation

# Evaluation only
python run_pipeline.py --skip-preprocessing --skip-training
```

### 3. Real-time Inference

#### Sign Language Recognition Only
```bash
python src/utils/webcam_inference.py --model models/finetuned/final_model.h5
```

#### Integrated Recognition (Sign Language + Emotion)
```bash
python src/utils/integrated_recognition.py \
    --sl-model models/finetuned/final_model.h5 \
    --emotion-model models/emotion_models.h5
```

#### Emotion Detection Only
```bash
python src/utils/emotion_demo.py --webcam
```

### 4. Batch Inference

```bash
# Evaluate on test dataset
python src/utils/run_inference.py --model models/finetuned/final_model.h5 --evaluate

# Predict single sample
python src/utils/run_inference.py --model models/finetuned/final_model.h5 --sample 0
```

### 5. Model Fine-tuning

```bash
python src/utils/partial_finetune.py
```

## 🔧 Configuration

Key configuration parameters in `src/config.py`:

- **Dataset**: Update `DATASET_ROOT` to point to your dataset
- **Model**: Adjust `IMAGE_SIZE`, `MAX_SEQUENCE_LENGTH`, `LSTM_UNITS`
- **Training**: Modify `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`
- **Inference**: Set `CAMERA_ID`, `CONFIDENCE_THRESHOLD`

## 📊 Model Architecture

The system uses a hybrid CNN-LSTM architecture:

1. **CNN Feature Extractor**: ResNet-style architecture with residual connections
2. **Temporal Modeling**: Bidirectional LSTM layers for sequence understanding
3. **Attention Mechanism**: Temporal attention to focus on important frames
4. **Classification**: Dense layers with dropout for final prediction

### Key Features:
- **Residual Connections**: Improved gradient flow
- **Bidirectional LSTM**: Better temporal context
- **Temporal Attention**: Focus on important frames
- **Test-Time Augmentation**: Enhanced inference accuracy

## 📈 Performance

The model achieves:
- **Test Accuracy**: ~85-90% on the ISL_CSLRT_Corpus dataset
- **Real-time Performance**: 10-15 FPS on modern hardware
- **Robustness**: Handles lighting variations and camera movements

## 🎮 Controls

### Webcam Interface
- **'q'**: Quit the application
- **'r'**: Start/stop recording gesture sequence
- **Real-time display**: Shows predictions and confidence scores

### Recording Mode
- Automatically saves gesture sequences to `recorded_gestures/`
- Configurable recording duration (default: 3 seconds)
- Supports manual start/stop

## 🔍 Troubleshooting

### Common Issues

1. **Camera not working**
   ```bash
   python src/utils/test_webcam.py
   ```
   - Check camera permissions in System Preferences
   - Try different camera indices (0, 1, 2)

2. **CUDA/GPU issues**
   - Ensure TensorFlow GPU version is installed
   - Check CUDA compatibility
   - Use CPU-only mode if needed

3. **Memory issues**
   - Reduce `BATCH_SIZE` in config
   - Use smaller `IMAGE_SIZE`
   - Enable gradient accumulation

4. **Dataset path issues**
   - Update `DATASET_ROOT` in `src/config.py`
   - Ensure dataset follows expected structure

### Performance Optimization

- **Training**: Use GPU with sufficient VRAM
- **Inference**: Reduce image size for faster processing
- **Real-time**: Adjust prediction intervals in config

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- ISL_CSLRT_Corpus dataset providers
- OpenCV and TensorFlow communities
- Research papers on temporal attention mechanisms

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue with detailed information

---

**Note**: This system is designed for research and educational purposes. For production use, additional testing and validation are recommended. 