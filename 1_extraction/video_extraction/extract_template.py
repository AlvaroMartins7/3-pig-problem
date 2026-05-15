import os
import json
import random
from PIL import Image


def has_intersection(rect1, rect2):
    """Check whether two center-format bounding boxes overlap.

    Each bbox dict must contain 'x', 'y' (center), 'width', and 'height'.
    Returns True if the rectangles intersect, False otherwise.
    """
    x1_min, y1_min, x1_max, y1_max = rect1['x'] - rect1['width'] / 2, rect1['y'] - rect1['height'] / 2, rect1['x'] + rect1['width'] / 2, rect1['y'] + rect1['height'] / 2
    x2_min, y2_min, x2_max, y2_max = rect2['x'] - rect2['width'] / 2, rect2['y'] - rect2['height'] / 2, rect2['x'] + rect2['width'] / 2, rect2['y'] + rect2['height'] / 2
    
    return not (x1_max <= x2_min or x1_min >= x2_max or y1_max <= y2_min or y1_min >= y2_max)

# --- Configuration ---
frames_dir = './output/rec_b/frames'  # Directory containing the extracted video frames
annotations_path = 'frames.json'       # Path to the VoTT-exported JSON annotation file
output_dir = 'annotated_parts'         # Output directory for cropped non-overlapping regions

num_frames_to_process = 100  # Total number of frames to randomly sample
num_partitions = 5           # Number of output partitions to split results into

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load bounding-box annotations from the JSON file
with open(annotations_path, 'r') as f:
    annotations = json.load(f)

# Group all bounding boxes by their frame number for overlap detection
conflict_regions = {}

for obj in annotations['objects']:
    for frame_data in obj['frames']:
        frame_number = frame_data['frameNumber']
        bbox = frame_data['bbox']

        if frame_number not in conflict_regions:
            conflict_regions[frame_number] = []
        conflict_regions[frame_number].append(bbox)

# Randomly sample frames (without replacement) for processing
all_frame_numbers = list(conflict_regions.keys())
selected_frame_numbers = random.sample(all_frame_numbers, min(num_frames_to_process, len(all_frame_numbers)))

# Split selected frames into equal-sized partitions for organized output
partition_size = len(selected_frame_numbers) // num_partitions
partitions = [selected_frame_numbers[i * partition_size:(i + 1) * partition_size] for i in range(num_partitions)]

# Process each partition of frames
for partition_idx, frame_numbers in enumerate(partitions, start=1):
    # Create a subdirectory for each partition
    partition_dir = os.path.join(output_dir, f'partition_{partition_idx}')
    if not os.path.exists(partition_dir):
        os.makedirs(partition_dir)

    for frame_number in frame_numbers:
        bboxes = conflict_regions[frame_number]
        conflicting_indices = set()

        # Detect overlapping bounding boxes within the same frame
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                if has_intersection(bboxes[i], bboxes[j]):
                    conflicting_indices.add(i)
                    conflicting_indices.add(j)

        # Load the corresponding frame image
        frame_filename = os.path.join(frames_dir, f'frame_{frame_number:04d}.jpg')
        if not os.path.exists(frame_filename):
            print(f"Frame {frame_filename} not found, skipping.")
            continue

        with Image.open(frame_filename) as img:
            # Crop and save only non-overlapping regions as templates
            for idx, bbox in enumerate(bboxes):
                if idx not in conflicting_indices:
                    x_center = bbox['x']
                    y_center = bbox['y']
                    width = bbox['width']
                    height = bbox['height']

                    # Convert center-format bbox to corner coordinates for cropping
                    left = x_center - width / 2
                    top = y_center - height / 2
                    right = x_center + width / 2
                    bottom = y_center + height / 2

                    cropped_img = img.crop((left, top, right, bottom))

                    output_filename = os.path.join(partition_dir, f'frame_{frame_number:04d}_region_{idx}.jpg')
                    cropped_img.save(output_filename)

                    print(f"Valid region saved to: {output_filename}")

print("Extraction complete. Overlapping regions were skipped.")
