"""Convert VoTT JSON annotations to YOLO-format text files.

Reads bounding-box annotations from a VoTT-exported JSON file and writes
one YOLO-format .txt file per frame containing only ground-truth annotations.
Each line in the output follows the format: class_id x_center y_center width height
(all values normalized to [0, 1]).
"""

import json
import os

# --- Configuration ---
# Frame resolution must match the source video (used for coordinate normalization)
image_width = 3840
image_height = 2160

json_path = "frames.json"       # Path to the VoTT-exported JSON annotation file
output_dir = "yolo_annotations"  # Output directory for YOLO-format annotation files

os.makedirs(output_dir, exist_ok=True)

# Load annotations from JSON
with open(json_path, "r") as f:
    data = json.load(f)

# Process each annotated object and its per-frame bounding boxes
for obj in data["objects"]:
    for frame in obj["frames"]:
        frame_number = frame["frameNumber"]
        bbox = frame["bbox"]
        is_ground_truth = frame.get("isGroundTruth", "0")

        # Only convert manually verified (ground-truth) annotations
        if is_ground_truth != "1":
            continue

        # Normalize center-format bbox coordinates to [0, 1] range for YOLO
        x_center = bbox["x"] / image_width
        y_center = bbox["y"] / image_height
        width = bbox["width"] / image_width
        height = bbox["height"] / image_height

        # Class ID 0 = pig (single-class detection)
        class_id = 0
        yolo_annotation = f"{class_id} {x_center} {y_center} {width} {height}\n"

        # Append annotation to the per-frame output file
        output_file = os.path.join(output_dir, f"frame_{frame_number:04d}.txt")

        with open(output_file, "a") as out_f:
            out_f.write(yolo_annotation)

print(f"YOLO annotations saved to: {output_dir}")
