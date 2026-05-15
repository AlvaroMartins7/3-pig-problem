"""Refine object templates using SAM2 (Segment Anything Model 2).

Loads rough-cropped templates from the extraction step, applies SAM2 point-based
segmentation using the image center as the prompt, and saves clean RGBA templates
with transparent backgrounds.
"""

import os
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import enhancing_tools as et  # Custom module with mask-application and cropping utilities

# --- Configuration ---
input_folder_path = "./data/"
output_folder_path = "./output/"

# Set device: use "cuda" for NVIDIA GPUs, "cpu" otherwise
device = torch.device("cpu")
print(f"using device: {device}")

# SAM2 model checkpoint and configuration paths
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Build and initialize the SAM2 predictor
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# Process each template image in the input folder
for filename in os.listdir(input_folder_path):
	image_path = os.path.join(input_folder_path,filename)
	print(f"Processing Image: {filename}")
	
	image = Image.open(image_path)
	image = np.array(image.convert("RGB"))

	# Use the image center as the SAM2 point prompt (assumes object is centered)
	image_width, image_height = image.shape[1], image.shape[0]
	center_x, center_y = (image_width // 2), (image_height // 2)

	predictor.set_image(image)

	# Provide a single positive point at the center of the template
	input_point = np.array([[center_x, center_y]])
	input_label = np.array([1])  # 1 = foreground point

	# Generate multiple mask predictions and select the best one
	masks, scores, logits = predictor.predict(
	    point_coords=input_point,
	    point_labels=input_label,
	    multimask_output=True,
	)
    
	# Sort masks by confidence score (descending) and pick the highest
	sorted_ind = np.argsort(scores)[::-1]
	masks = masks[sorted_ind]
	best_mask = masks[0]
	
	# Apply the mask as an alpha channel and crop to the object bounding box
	modified_image = et.apply_mask_and_remove(best_mask, image)
	cropped_image = et.crop_to_content(modified_image)
    
	# Save as PNG to preserve transparency
	output_image_path = os.path.join(output_folder_path, os.path.splitext(filename)[0] + ".png")
	Image.fromarray(cropped_image).save(output_image_path)
	print(f"Saved processed image to {output_image_path}")