"""Visualization utility to verify bounding-box annotations on a single frame.

Draws all annotated bounding boxes for a given frame number and displays the result.
Useful for spot-checking annotation quality before running the full extraction pipeline.
"""

import cv2
import json
import os
from PIL import Image

# --- Configuration ---
frames_dir = './output/2024-11-06_15-31-30/frames'  # Directory containing extracted frames
annotations_path = 'frames.json'                     # VoTT-exported JSON annotation file
output_dir = 'tested_frames'                         # Output directory for annotated frames
frame_to_process = 2050                              # Frame number to visualize (change as needed)

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load annotations from JSON
with open(annotations_path, 'r') as f:
    annotations = json.load(f)

frame_found = False

# Load the target frame image
frame_filename = os.path.join(frames_dir, f'frame_{frame_to_process:04d}.jpg')
image = cv2.imread(frame_filename)

if image is None:
    print(f"Frame {frame_filename} not found.")
else:
    frame_found = True

    # Draw all bounding boxes for this frame from every annotated object
    for obj in annotations['objects']:
        for frame_data in obj['frames']:
            if frame_data['frameNumber'] == frame_to_process:
                bbox = frame_data['bbox']
                
                # Bbox coordinates are center-format (x, y = center of box)
                x_center, y_center = bbox['x'], bbox['y']
                width, height = bbox['width'], bbox['height']
                
                # Convert center-format to corner coordinates for OpenCV
                top_left = (int(x_center - width / 2), int(y_center - height / 2))
                bottom_right = (int(x_center + width / 2), int(y_center + height / 2))

                color = (0, 255, 0)  # Green bounding boxes
                thickness = 5

                cv2.rectangle(image, top_left, bottom_right, color, thickness)

    # Convert from OpenCV BGR to RGB for Pillow display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Display the annotated frame
    pil_image.show()


if not frame_found:
    print(f"Frame {frame_to_process} was not found in the annotations.")