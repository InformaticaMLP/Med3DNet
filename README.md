# Medical Image Segmentation with 3D U-Net and ResNet

This project implements a deep learning pipeline for medical image segmentation using 3D U-Net with ResNet blocks and attention mechanisms.

## Project Structure

```
.
├── data_preprocessing.py   # DICOM processing and 3D data augmentation
├── model.py               # 3D U-Net + ResNet multi-task model with attention
├── train.py               # Training with combined loss (Dice + CrossEntropy)
├── evaluate.py            # Evaluation with Dice coefficient
├── inference.py           # DICOM inference and OpenCV visualization
├── config.py              # Centralized configuration management
└── README.md              # Project documentation
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- SimpleITK
- OpenCV
- medpy

Install requirements:
```bash
pip install torch SimpleITK opencv-python medpy
```

## Usage

### 1. Data Preparation
Place your DICOM images in the following structure:
```
data/
├── train/
│   ├── images/    # Training DICOM images
│   └── labels/    # Corresponding segmentation masks
└── val/
    ├── images/    # Validation DICOM images
    └── labels/    # Corresponding segmentation masks
```

### 2. Training
```bash
python train.py
```

### 3. Evaluation
```bash
python evaluate.py
```

### 4. Inference
```bash
python inference.py
```

## Configuration
Modify `config.py` to adjust:
- Model parameters (number of classes)
- Training hyperparameters (batch size, learning rate)
- Data paths
- Loss function weights

## Results
Evaluation reports are saved in `reports/` directory.
Segmentation results are saved in `results/` directory.

## License
MIT