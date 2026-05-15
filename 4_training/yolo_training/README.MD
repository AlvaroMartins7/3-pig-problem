# YOLO Training

## Overview

Trains a YOLOv11x object detection model on the synthetic pig detection dataset generated in Step 3, using the [Ultralytics](https://docs.ultralytics.com/) framework. After training, the script evaluates the model on the validation set, runs a quick inference test, and exports the model to ONNX format.

## How It Works

1. Loads a pre-trained `yolo11x.pt` model as the starting point.
2. Trains for 100 epochs on the dataset defined in `dataset.yaml`.
3. Evaluates performance on the validation split.
4. Runs inference on a test image for visual verification.
5. Exports the trained model to ONNX format for deployment.

## Configuration

`dataset.yaml` defines the dataset paths and class configuration:

```yaml
train: "./dataset/train/images"   # Training images directory
val: "./dataset/val/images"       # Validation images directory
nc: 1                             # Number of classes
names: ["0"]                      # Class names (single class: pig)
```

### Expected directory structure

```
yolo_training/
├── dataset/
│   ├── train/
│   │   ├── images/          # Training images from Step 3
│   │   └── labels/          # Corresponding YOLO annotation .txt files
│   └── val/
│       ├── images/          # Validation images
│       └── labels/          # Corresponding YOLO annotation .txt files
├── test/
│   └── test.jpg             # A sample image for quick inference verification
├── dataset.yaml
├── training_script.py
├── requirements.txt
└── README.MD
```

## Prerequisites

- Python 3.8+
- pip
- NVIDIA GPU with CUDA (recommended for training)

## Setup & Usage

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.\.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install ultralytics
```

> **Note:** The `ultralytics` package will automatically install all required sub-dependencies (PyTorch, OpenCV, etc.). The provided `requirements.txt` contains a full system-level `pip freeze` and is not the minimal dependency list.

### 4. Prepare the dataset

Organize your generated synthetic images (from Step 3) into `dataset/train/` and `dataset/val/` with matching `images/` and `labels/` subdirectories. Update `dataset.yaml` if your paths differ.

### 5. Run training

```bash
python training_script.py
```

Training results (weights, metrics, plots) will be saved under `runs/detect/train/` by default.

### 6. Customization

To modify training parameters, edit `training_script.py`:
- `epochs`: Number of training epochs (default: 100)
- `device`: GPU index or `"cpu"` (default: `"0"`)
- `data`: Path to the dataset YAML file
