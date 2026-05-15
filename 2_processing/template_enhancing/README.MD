# Template Enhancement with SAM2

## Overview

Refines rough-cropped object templates by applying [SAM2 (Segment Anything Model 2)](https://github.com/facebookresearch/sam2) segmentation. This step produces clean RGBA templates with transparent backgrounds, ready for compositing onto synthetic backgrounds in the generation step.

## How It Works

1. Loads each rough-cropped template image from the `data/` folder.
2. Uses the center of the image as a point prompt for SAM2.
3. SAM2 generates multiple segmentation masks; the highest-confidence mask is selected.
4. The mask is applied as an alpha channel, making the background transparent.
5. The image is cropped to the object bounding box and saved as a PNG with transparency.

## Scripts

| Script | Description |
|---|---|
| `enhance_templates.py` | Main script that loads templates, runs SAM2 segmentation, and saves enhanced RGBA templates. |
| `enhancing_tools.py` | Utility functions for applying segmentation masks and cropping to non-transparent content. |

### Expected directory structure

```
template_enhancing/
├── data/                    # Input: rough-cropped templates from Step 1
│   ├── template_001.jpg
│   ├── template_002.jpg
│   └── ...
├── output/                  # Output: clean RGBA templates (PNG)
├── checkpoints/
│   └── sam2.1_hiera_large.pt   # SAM2 model checkpoint
├── enhance_templates.py
├── enhancing_tools.py
├── requirements.txt
└── README.MD
```

## Prerequisites

- Python 3.8+
- pip
- (Optional) NVIDIA GPU with CUDA for faster inference

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
pip install torch torchvision pillow numpy
```

### 4. Install SAM2

```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

### 5. Download the SAM2 model checkpoint

Download the `sam2.1_hiera_large.pt` checkpoint from the [SAM2 repository](https://github.com/facebookresearch/sam2#model-checkpoints) and place it in a `checkpoints/` directory:

```bash
mkdir checkpoints
# Download sam2.1_hiera_large.pt into checkpoints/
```

### 6. Prepare the input data

Place the rough-cropped template images (output from Step 1) into the `data/` directory. Create the `output/` directory:

```bash
mkdir data output
```

### 7. Run the enhancement

```bash
python enhance_templates.py
```

> **Tip:** To use GPU acceleration, change `torch.device("cpu")` to `torch.device("cuda")` in `enhance_templates.py`.

Enhanced templates (PNG with transparency) will be saved to `output/`.
