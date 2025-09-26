# PPE Detection with YOLO11

A computer vision project for detecting Personal Protective Equipment (PPE) in workplace environments using YOLO11 (YOLOv11) object detection model.

## Overview

This project implements a custom-trained YOLO11 model to detect and classify various types of PPE including:
- **Helmets** - Safety head protection
- **Vests** - High-visibility safety vests
- **Boots** - Safety footwear
- **Person** - Workers in the scene

The model is trained on a specialized PPE dataset and achieves high accuracy for workplace safety monitoring applications.

## Features

- 🎯 **High Accuracy**: mAP50 of 0.98 and mAP50-95 of 0.792
- 🚀 **Real-time Detection**: Fast inference on both images and videos
- 🏭 **Workplace Safety**: Specifically designed for industrial safety monitoring
- 📊 **Comprehensive Analysis**: Includes confusion matrix and performance metrics
- 🎥 **Video Support**: Can process video streams for continuous monitoring

## Model Performance

The trained model shows excellent performance across all PPE categories:

| Class    | Precision | Recall | mAP50  | mAP50-95 |
|----------|-----------|--------|--------|----------|
| All      | 0.963     | 0.939  | 0.98   | 0.792    |
| Boots    | 0.98      | 0.897  | 0.977  | 0.725    |
| Helmet   | 0.974     | 0.956  | 0.977  | 0.737    |
| Person   | 0.925     | 0.935  | 0.977  | 0.875    |
| Vest     | 0.974     | 0.967  | 0.987  | 0.83     |

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Dependencies

```bash
pip install ultralytics
pip install roboflow
```

## Usage

### 1. Dataset Setup

The project uses the PPE Dataset for Workplace Safety from Roboflow:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="your_api_key")
project = rf.workspace("siabar").project("ppe-dataset-for-workplace-safety")
version = project.version(2)
dataset = version.download("yolov11")
```

### 2. Training

Train the YOLO11 model on the PPE dataset:

```bash
yolo task=detect mode=train model="yolo11n.pt" data=path/to/data.yaml epochs=50 imgsz=640
```

### 3. Validation

Validate the trained model:

```bash
yolo task=detect mode=val model="path/to/best.pt" data=path/to/data.yaml
```

### 4. Inference

#### On Images

```bash
yolo task=detect mode=predict model="path/to/best.pt" conf=0.25 source=image.jpg save=True
```

#### On Videos

```bash
yolo task=detect mode=predict model="path/to/best.pt" conf=0.25 source=video.mp4 save=True
```

## Project Structure

```
project/
├── custom_training.ipynb    # Main training notebook
├── README.md               # This file
└── runs/                   # Training outputs
    └── detect/
        ├── train2/         # Training results
        │   ├── weights/    # Model weights
        │   ├── confusion_matrix.png
        │   ├── results.png
        │   └── labels.jpg
        └── predict*/       # Prediction results
```

## Training Details

- **Model**: YOLO11n (nano version)
- **Epochs**: 50
- **Image Size**: 640x640
- **Batch Size**: 16
- **Optimizer**: Auto (AdamW)
- **Learning Rate**: 0.01
- **Dataset Split**: Train/Validation/Test

## Results Analysis

### Confusion Matrix Insights

The model shows some confusion between classes:
- **Boots**: 38 instances predicted as background
- **Helmet**: 7 instances predicted as background
- **Person**: 9 instances predicted as background
- **Vest**: 6 instances predicted as background, 2 as person

### Performance Metrics

- **Overall mAP50**: 0.98 (excellent)
- **Overall mAP50-95**: 0.792 (very good)
- **Inference Speed**: ~7-12ms per image
- **Model Size**: 2.6M parameters (lightweight)

## Applications

This model can be used for:

- 🏗️ **Construction Site Monitoring**: Ensure workers wear proper PPE
- 🏭 **Industrial Safety**: Automated safety compliance checking
- 📹 **Video Surveillance**: Real-time PPE detection in security feeds
- 📊 **Safety Analytics**: Track PPE usage patterns and compliance rates
- 🚨 **Alert Systems**: Trigger notifications when PPE violations are detected

## Future Improvements

- [ ] Add more PPE classes (gloves, safety glasses, etc.)
- [ ] Implement real-time video streaming
- [ ] Add confidence threshold tuning
- [ ] Create web interface for easy deployment
- [ ] Add data augmentation techniques
- [ ] Implement model quantization for edge deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for educational and research purposes. Please ensure you have proper permissions when using the dataset and model for commercial applications.

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for the YOLO11 implementation
- [Roboflow](https://roboflow.com/) for the PPE dataset
- The open-source computer vision community

## Contact

For questions or support, please open an issue in the repository.
