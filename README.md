# Synthetic Training Data Generation for Pig Detection

A pipeline for generating synthetic training data for computer vision object detection models. The system extracts object templates from manually annotated video frames, refines them using SAM2 segmentation, composites them onto backgrounds with augmentations, and trains/evaluates YOLO models on the resulting dataset.

## Pipeline Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. Extraction   │────▶│  2. Processing   │────▶│  3. Generation   │────▶│   4. Training    │────▶│  5. Evaluation   │
│                 │     │                 │     │                 │     │                 │     │                 │
│ Extract object  │     │ Refine templates │     │ Generate synth  │     │ Train YOLO      │     │ Evaluate model  │
│ templates from  │     │ with SAM2       │     │ images + YOLO   │     │ detection model  │     │ on test data    │
│ annotated data  │     │ segmentation    │     │ annotations     │     │                 │     │ and video       │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Project Structure

```
3-pig-problem/
├── 1_extraction/                    # Step 1: Template extraction from annotated data
│   ├── template_extraction/         #   Extract templates from CSV-annotated images (VIA format)
│   │   ├── extract_templates.py
│   │   ├── requirements.txt
│   │   └── README.MD
│   └── video_extraction/            #   Extract frames and templates from annotated videos (VoTT format)
│       ├── extract_frames.py        #     Video → individual frames
│       ├── extract_template.py      #     Frames + JSON annotations → cropped templates
│       ├── video_annotation.py      #     JSON annotations → YOLO format labels
│       ├── test_frame.py            #     Visualization utility for annotation QA
│       ├── config.yaml
│       ├── frames.json
│       ├── requirements.txt
│       └── README.MD
│
├── 2_processing/                    # Step 2: Template refinement
│   └── template_enhancing/          #   SAM2-based background removal
│       ├── enhance_templates.py     #     Main segmentation script
│       ├── enhancing_tools.py       #     Mask application & cropping utilities
│       ├── requirements.txt
│       └── README.MD
│
├── 3_generation/                    # Step 3: Synthetic image generation
│   ├── README.MD
│   └── synth_img/
│       ├── generate_img.py          #     Main generator with multiprocessing support
│       ├── synth_tools.py           #     Image manipulation & augmentation utilities
│       ├── config.yaml              #     Generation parameters
│       ├── requirements.txt
│       └── README.MD
│
├── 4_training/                      # Step 4: Model training
│   └── yolo_training/
│       ├── training_script.py       #     YOLOv11x training, validation & ONNX export
│       ├── dataset.yaml             #     Dataset configuration
│       ├── requirements.txt
│       └── README.MD
│
├── 5_evaluation/                    # Step 5: Model evaluation
│   ├── eval.py                      #     Quantitative metrics (mAP, precision, recall)
│   ├── video_test.py                #     Real-time video inference visualization
│   ├── dataset.yaml                 #     Evaluation dataset configuration
│   ├── requirements.txt
│   └── README.MD
│
└── README.md                        # This file
```

## Step-by-Step Reproduction Guide

### Step 1: Template Extraction

Extract object templates from manually annotated image/video datasets. Two extraction methods are available:

- **From image datasets (VIA CSV annotations):** Use `1_extraction/template_extraction/` to extract templates from the [AIFARMS pig tracking dataset](https://drive.google.com/drive/folders/1E2wW2aRENgy_TqlzfICn58ahbTHVIaK6).
- **From video files (VoTT JSON annotations):** Use `1_extraction/video_extraction/` to first extract video frames, then crop templates using bounding-box annotations.

Both methods detect and skip overlapping bounding boxes to ensure clean, isolated templates.

See: [`1_extraction/template_extraction/README.MD`](1_extraction/template_extraction/README.MD) | [`1_extraction/video_extraction/README.MD`](1_extraction/video_extraction/README.MD)

### Step 2: Template Enhancement with SAM2

Refine rough-cropped templates using SAM2 segmentation to remove backgrounds. This produces clean RGBA images with transparent backgrounds, ensuring that only the object (pig) pixels are preserved.

**Requirements:** SAM2 model checkpoint (`sam2.1_hiera_large.pt`). GPU recommended.

See: [`2_processing/template_enhancing/README.MD`](2_processing/template_enhancing/README.MD)

### Step 3: Synthetic Image Generation

Generate a complete training dataset by compositing enhanced templates onto backgrounds with configurable augmentations:

- Random placement, scaling, rotation, flipping
- Gaussian blur and edge-blending for realistic compositing
- Global brightness, contrast, sharpness, and color jitter
- Noise backgrounds or real background images
- Automatic YOLO-format annotation generation
- Parallel processing for fast generation

See: [`3_generation/README.MD`](3_generation/README.MD)

### Step 4: YOLO Model Training

Train a YOLOv11x detection model on the synthetic dataset using the Ultralytics framework. The training script handles the full workflow: training, validation, test inference, and ONNX export.

**Requirements:** NVIDIA GPU with CUDA recommended.

See: [`4_training/yolo_training/README.MD`](4_training/yolo_training/README.MD)

### Step 5: Model Evaluation

Evaluate the trained model quantitatively (mAP, precision, recall) and qualitatively (real-time video inference). Metrics are exported to JSON for analysis.

See: [`5_evaluation/README.MD`](5_evaluation/README.MD)

## Dataset

The original annotated dataset was produced by [AIFARMS](https://github.com/AIFARMS/multi-camera-pig-tracking) and is available on [Google Drive](https://drive.google.com/drive/folders/1E2wW2aRENgy_TqlzfICn58ahbTHVIaK6). Download the `ISRL images` directory for image-based extraction, or use your own annotated video data for video-based extraction.

## General Requirements

- Python 3.8+
- pip
- (Steps 2, 4, 5) NVIDIA GPU with CUDA is strongly recommended

Each step has its own dependencies documented in its respective README. The core libraries used across the project are:

| Library | Used In | Purpose |
|---|---|---|
| `Pillow` | Steps 1, 2, 3 | Image loading, cropping, manipulation |
| `opencv-python` | Steps 1, 3, 5 | Video processing, image operations |
| `numpy` | Steps 2, 3, 5 | Array operations |
| `pandas` | Step 1 | CSV annotation parsing |
| `PyYAML` | Steps 1, 3 | Configuration file parsing |
| `torch` / `torchvision` | Step 2 | SAM2 model inference |
| `sam2` | Step 2 | Segment Anything Model 2 |
| `imgaug` | Step 3 | Image augmentation (Gaussian blur) |
| `ultralytics` | Steps 4, 5 | YOLO model training and inference |

> **Note on requirements.txt files:** The `requirements.txt` files in each subdirectory contain full system-level `pip freeze` outputs rather than minimal dependency lists. Refer to the individual README files for the actual required packages.
