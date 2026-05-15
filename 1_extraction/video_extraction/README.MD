# Video Frame Extraction & Annotation Tools

## Overview

A collection of scripts to extract frames from video files, crop object templates from those frames using JSON annotations (VoTT format), convert annotations to YOLO format, and visually verify annotations.

## Scripts

| Script | Description |
|---|---|
| `extract_frames.py` | Extracts every frame from a video file and saves them as numbered JPEG images. |
| `extract_template.py` | Crops non-overlapping object templates from extracted frames using VoTT JSON annotations, partitioning output into subsets. |
| `video_annotation.py` | Converts VoTT JSON annotations to YOLO-format `.txt` label files (one per frame, only ground-truth boxes). |
| `test_frame.py` | Visualization utility that draws all bounding boxes on a specific frame for quality checking. |

## Configuration

`config.yaml` controls paths for `extract_frames.py`:

```yaml
video_path: "./data"      # Directory containing source video files
output_path: "./output"   # Base output directory for extracted frames
```

The JSON annotation file (`frames.json`) should be a VoTT export with center-format bounding boxes (`x`, `y` = center, `width`, `height`).

### Expected directory structure

```
video_extraction/
├── data/
│   └── filmagem.mp4        # Source video file
├── output/                  # Created automatically with timestamped subdirectories
├── config.yaml
├── frames.json              # VoTT-exported annotation file
├── extract_frames.py
├── extract_template.py
├── video_annotation.py
├── test_frame.py
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
pip install opencv-python pillow pyyaml
```

> **Note:** The provided `requirements.txt` contains a full system-level `pip freeze`. The only actual dependencies are `opencv-python`, `Pillow`, and `PyYAML`.

### 4. Extract frames from a video

```bash
python extract_frames.py --config config.yaml --video filmagem.mp4
```

This creates a timestamped folder under `./output/` containing all extracted frames.

### 5. Extract templates from annotated frames

Edit the paths at the top of `extract_template.py` to point to your extracted frames directory and annotation file, then run:

```bash
python extract_template.py
```

Templates are saved into numbered partition subdirectories under `annotated_parts/`.

### 6. Convert annotations to YOLO format

Edit `video_annotation.py` to set the correct `image_width` and `image_height` for your video, then run:

```bash
python video_annotation.py
```

YOLO-format `.txt` files are saved to `yolo_annotations/`.

### 7. Visualize annotations on a frame (optional)

Edit `test_frame.py` to set the desired `frame_to_process` number and correct `frames_dir`, then run:

```bash
python test_frame.py
```

This opens a window showing the frame with green bounding boxes drawn on it.
