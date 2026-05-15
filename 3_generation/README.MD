# Synthetic Image Generation

## Overview

Generates synthetic training images for YOLO object detection by compositing object templates onto background images (or random noise). Produces both the composite images and corresponding YOLO-format annotation files. Supports configurable augmentations and parallel processing.

## How It Works

1. Selects a background: either a random image from a backgrounds folder or a generated noise image.
2. Randomly selects 1 to N templates, and for each:
   - Applies a filtering pipeline (Gaussian blur + multi-layer edge blending) for realistic compositing.
   - Normalizes the template to match the target resolution.
   - Applies random rescaling, horizontal/vertical flipping, and rotation.
   - Places the template at a random non-overlapping position on the background.
   - Records the bounding box in YOLO annotation format.
3. Applies global image augmentations: brightness, contrast, sharpness, and color jitter.
4. Saves the final composite image and its annotation file.

## Scripts

| Script | Description |
|---|---|
| `generate_img.py` | Main entry point. Parses arguments, loads config, and orchestrates parallel image generation. |
| `synth_tools.py` | Utility library with all image manipulation, augmentation, annotation, and compositing functions. |

## Configuration

All generation parameters are configured via `config.yaml`:

| Parameter | Description |
|---|---|
| `bgns_path` | Path to background images directory |
| `tpls_path` | Path to template images directory (RGBA PNGs from Step 2) |
| `out_path` | Base output directory |
| `noise_bg` | Probability of using a noise background instead of a real image (0.0–1.0) |
| `img_res` | Output image resolution as `[width, height]` |
| `orig_tpl_res` | Resolution of the original images the templates were cropped from |
| `rescale_factor` | Random rescale range (e.g., 0.05 means ±5%) |
| `rotation_angle` | Maximum random rotation angle in degrees |
| `h_flip` | Probability of horizontal flip (0.0–1.0) |
| `v_flip` | Probability of vertical flip (0.0–1.0) |
| `templ_number` | Maximum number of templates to place per image |
| `gamma_factor` | Random brightness adjustment range |
| `contrast_factor` | Random contrast adjustment range |
| `sharpness_factor` | Random sharpness adjustment range |
| `color_factor` | Random color saturation adjustment range |

### Expected directory structure

```
synth_img/
├── data/
│   ├── backgrounds/         # Background images (any format)
│   └── templates/           # RGBA template PNGs (from Step 2)
├── output/                  # Created automatically with timestamped subdirectories
│   └── YYYY-MM-DD_HH-MM-SS/
│       ├── images/          # Generated composite images
│       └── labels/          # YOLO-format annotation .txt files
├── config.yaml
├── generate_img.py
├── synth_tools.py
├── requirements.txt
└── README.MD
```

## Prerequisites

- Python 3.8+
- pip

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
pip install pillow opencv-python numpy pyyaml imgaug
```

> **Note:** The provided `requirements.txt` contains a full system-level `pip freeze`. The actual dependencies are `Pillow`, `opencv-python`, `numpy`, `PyYAML`, and `imgaug`.

### 4. Prepare the data

- Place background images in `data/backgrounds/`.
- Place enhanced RGBA template PNGs (from Step 2) in `data/templates/`.

### 5. Generate the dataset

**Basic usage:**
```bash
python generate_img.py --config config.yaml --num-imgs 100 --max-process 4
```

**With debug mode** (draws bounding boxes on images for visual verification):
```bash
python generate_img.py --config config.yaml --num-imgs 10 --max-process 1 --debug
```

### Command-line arguments

| Argument | Required | Description |
|---|---|---|
| `--config` | Yes | Path to the YAML configuration file |
| `--num-imgs` | Yes | Number of synthetic images to generate |
| `--max-process` | No (default: 1) | Number of parallel worker processes |
| `--debug` | No | Enable debug mode to draw bounding boxes on images |

Output images and labels are saved in a timestamped directory under `output/`.
