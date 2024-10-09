import os
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import enhancing_tools as et  # my custom module

input_folder_path = "./data/"
output_folder_path = "./output/"

# "cuda" se tiver gpu da Nvidia, "cpu" se não tiver
device = torch.device("cpu")
print(f"using device: {device}")

sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

for filename in os.listdir(input_folder_path):
	image_path = os.path.join(input_folder_path,filename)
	print(f"Processing Image: {filename}")
	
	image = Image.open(image_path)
	image = np.array(image.convert("RGB"))

	image_width, image_height = image.shape[1], image.shape[0]
	center_x, center_y = (image_width // 2), (image_height // 2)

	predictor.set_image(image)

	input_point = np.array([[center_x, center_y]])
	input_label = np.array([1])

	masks, scores, logits = predictor.predict(
	    point_coords=input_point,
	    point_labels=input_label,
	    multimask_output=True,
	)
    
	sorted_ind = np.argsort(scores)[::-1]
	masks = masks[sorted_ind]
	best_mask = masks[0]	# Select the mask with the highest score
	
	modified_image = et.apply_mask_and_remove(best_mask, image)
	cropped_image = et.crop_to_content(modified_image)
    
	output_image_path = os.path.join(output_folder_path, os.path.splitext(filename)[0] + ".png")
	Image.fromarray(cropped_image).save(output_image_path)
	print(f"Saved processed image to {output_image_path}")