# Template Extraction from Image Annotations

## Overview

Extracts individual object templates (cropped images) from annotated image datasets. The script reads CSV annotations exported from [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/), detects overlapping bounding boxes, and only extracts non-overlapping regions to ensure clean, isolated templates.

## How It Works

1. Parses the CSV annotation file containing bounding-box coordinates for each image.
2. For each image, compares all annotated regions pairwise to detect overlaps.
3. Discards any region involved in an overlap (to avoid templates with partial objects).
4. Crops and saves each valid (non-overlapping) region as an individual JPEG file.

## Dataset

The dataset used was produced by [AIFARMS](https://github.com/AIFARMS/multi-camera-pig-tracking) and can be accessed via this [Google Drive link](https://drive.google.com/drive/folders/1E2wW2aRENgy_TqlzfICn58ahbTHVIaK6).

1. Download the `ISRL images` directory.
2. Place a subdirectory containing images and annotations (e.g., `12-00/`) inside a `data/` folder in this directory.

### Expected directory structure

```
template_extraction/
├── data/
│   └── 12-00/
│       ├── 12-00.csv          # VIA annotation export
│       ├── image_001.jpg
│       ├── image_002.jpg
│       └── ...
├── output/                    # Created automatically
├── extract_templates.py
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
pip install pandas pillow
```

> **Note:** The provided `requirements.txt` contains a full system-level `pip freeze`. The only actual dependencies for this script are `pandas` and `Pillow`.

### 4. Prepare the dataset

Make sure the `data/12-00/` directory exists and contains the images and CSV annotation file. Adjust the path inside `extract_templates.py` if your data is in a different location.

### 5. Create the output directory

```bash
mkdir output
```

### 6. Run the extraction

```bash
python extract_templates.py
```

Extracted templates will be saved to the `output/` directory.
