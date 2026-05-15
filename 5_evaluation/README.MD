# Model Evaluation

## Overview

Evaluates trained YOLO model weights on a test dataset and provides both quantitative metrics and real-time video inference visualization.

## Scripts

| Script | Description |
|---|---|
| `eval.py` | Evaluates the model on the test split, computes mAP50, mAP50-95, precision, and recall, and saves results to `metrics_results.json`. |
| `video_test.py` | Runs real-time object detection on a video file and displays detections in an OpenCV window. Press `q` to quit. |

## Configuration

`dataset.yaml` defines the dataset splits for evaluation:

```yaml
train: "<path>/dataset/val_bg/images"      # Training images (background-based)
val: "<path>/dataset/val_noise/images"      # Validation images (noise-based)
test: "<path>/dataset/test/images"          # Test images for final evaluation
nc: 1                                       # Number of classes
names: ["0"]                                # Class names
```

> **Important:** Update the absolute paths in `dataset.yaml` to match your local environment.

### Expected directory structure

```
5_evaluation/
├── dataset/
│   ├── val_bg/
│   │   ├── images/
│   │   └── labels/
│   ├── val_noise/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── experimentos/                # Trained model weights organized by experiment
│   └── results_2/
│       └── 60_art/
│           └── detect/train/weights/best.pt
├── 12-Ceiling_Cam.mp4           # Test video for real-time inference
├── dataset.yaml
├── eval.py
├── video_test.py
├── requirements.txt
└── README.MD
```

## Prerequisites

- Python 3.8+
- pip
- NVIDIA GPU with CUDA (recommended)

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
pip install ultralytics numpy
```

> **Note:** The provided `requirements.txt` contains a full system-level `pip freeze`. The actual dependencies are `ultralytics` and `numpy`.

### 4. Run quantitative evaluation

Edit `eval.py` to point to your trained model weights, then:

```bash
python eval.py
```

Metrics are printed to the console and saved to `metrics_results.json`:
- **mAP50**: Mean Average Precision at IoU 0.50
- **mAP50-95**: Mean Average Precision at IoU 0.50–0.95
- **Precision**: Mean precision across all classes
- **Recall**: Mean recall across all classes

### 5. Run video inference test

Edit `video_test.py` to point to your trained model weights and video file, then:

```bash
python video_test.py
```

A window will open showing real-time detections on the video. Press `q` to exit.
