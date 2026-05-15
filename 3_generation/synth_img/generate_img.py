"""Synthetic image dataset generator for YOLO object detection training.

Generates composite images by placing randomly transformed object templates onto
background images (or noise backgrounds), and produces corresponding YOLO-format
annotation files. Supports multiprocessing for parallel generation.
"""

import os
import argparse
import yaml
import random
import datetime as dt
from concurrent.futures import ProcessPoolExecutor 
from PIL import ImageDraw

import synth_tools as st  # Custom module with image manipulation utilities


def write_annotation(annot_folder, img_filename, data_array):
    """Append a single YOLO-format annotation line to the corresponding label file."""

    file_path = os.path.join(annot_folder, f"{img_filename}.txt")
    with open(file_path, 'a') as file:
        data_line = ' '.join(map(str, data_array))
        file.write(data_line + '\n')


def process_template(background, template_folder, annot_folder, img_filename, rescale_factor, 
					 debug, rotation_angle, h_flip, v_flip, norm_factor):
	"""Load a random template, apply augmentations, place it on the background, and record the annotation.

	Returns True if the template was successfully placed, False if it was too large to fit.
	"""
	    
	template = st.get_random_image(template_folder).convert('RGBA')

	# Apply augmentation pipeline: blur+blend filter, rescale, normalize, flip, rotate
	template = st.filter_template(template)
	template = st.rescale_img(template, rescale_factor)
	template = st.normalize_img(template, norm_factor)
	template = st.flip_img(template, h_flip, v_flip)
	template = st.rotate_img(template, rotation_angle)

	position = st.get_random_position(background, template)
	if position is None:
		return False

	st.overlay_template(background, template, position)
	bbox = st.calculate_bounding_box(position, template)
    
	yolo_annot_params = st.set_yolo_annot_params(bbox[0], bbox[1], bbox[2], bbox[3], background.size[0], background.size[1], 0)
	write_annotation(annot_folder, img_filename, yolo_annot_params)

	# In debug mode, draw visible bounding boxes on the image for visual inspection
	if debug:
		draw = ImageDraw.Draw(background)
		draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="yellow", width=3)
    
	# 30% chance to paste the template again (simulates partial occlusion / double-exposure)
	if random.random() < 0.3:
		st.overlay_template(background, template, position)
    
	return True


def merge_images(background_folder, template_folder, annot_folder, img_filename, img_res, debug,
				 noise_bg, rescale_factor, rotation_angle, h_flip, v_flip, templ_number, gamma_factor,
				 contrast_factor, sharpness_factor, color_factor, norm_factor):
	"""Compose a single synthetic image with multiple templates on a background.

	Selects a noise or real background, places a random number of augmented templates,
	and applies global color/brightness/contrast/sharpness augmentations.
	"""
	
	# Select background: random noise or a real image from the backgrounds folder
	if random.random() < noise_bg:
		background = st.noise_background(img_res).convert('RGBA')
	else:
		background = st.get_random_image(background_folder).convert('RGBA')
		background = st.resize_img(background, img_res)
	
	# Place a random number of templates (between 1 and templ_number)
	num_templates = random.randint(1, templ_number)
	for _ in range(num_templates):
		success = process_template(background, template_folder, annot_folder, img_filename, rescale_factor, 
							 	   debug, rotation_angle, h_flip, v_flip, norm_factor)
		if not success:
			continue
	
	# Apply global image augmentations to the entire composite
	background = st.gamma_transform(background, gamma_factor)
	background = st.contrast_transform(background, contrast_factor)
	background = st.sharpness_transform(background, sharpness_factor)
	background = st.color_transform(background, color_factor)

	return background


def generate_image(index, bgns_path, tpls_path, image_path, annotation_path, img_res,
				   debug, noise_bg, rescale_factor, rotation_angle, h_flip, v_flip, templ_number,
				   gamma_factor, contrast_factor, sharpness_factor, color_factor, norm_factor):
	"""Generate a single synthetic image with annotations and save both to disk."""
	
	img_filename = f"image_{index:03d}"
	synth_img = merge_images(bgns_path, tpls_path, annotation_path, img_filename, img_res, debug,
							 noise_bg, rescale_factor, rotation_angle, h_flip, v_flip, templ_number, 
							 gamma_factor, contrast_factor, sharpness_factor, color_factor, norm_factor)
	output_path = os.path.join(image_path, f"{img_filename}.png")
	synth_img.save(output_path, format='PNG')
	print(f"{img_filename} and labels were generated successfully.")


def create_folder(out_path):
	"""Create a timestamped output directory with 'images' and 'labels' subdirectories.

	Returns a tuple of (image_path, annotation_path).
	"""

	actual_dt = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	out_path = os.path.join(out_path, actual_dt)
	image_path = os.path.join(out_path, 'images')
	annotation_path = os.path.join(out_path, 'labels')

	os.makedirs(image_path)
	os.makedirs(annotation_path)

	return image_path, annotation_path


def generate_dataset(bgns_path, tpls_path, out_path, num_imgs, max_process, img_res,
				  	 debug, noise_bg, rescale_factor, rotation_angle, h_flip, v_flip, 
					 templ_number, gamma_factor, contrast_factor, sharpness_factor, 
					 color_factor, norm_factor):
	"""Generate a full synthetic dataset using multiprocessing for parallel image creation."""
	
	image_path, annotation_path = create_folder(out_path)

	with ProcessPoolExecutor(max_workers=max_process) as executor:
		executor.map(generate_image, range(num_imgs), [bgns_path]*num_imgs, [tpls_path]*num_imgs, 
                     [image_path]*num_imgs, [annotation_path]*num_imgs, [img_res]*num_imgs,
					 [debug]*num_imgs, [noise_bg]*num_imgs, [rescale_factor]*num_imgs, [rotation_angle]*num_imgs,
					 [h_flip]*num_imgs, [v_flip]*num_imgs, [templ_number]*num_imgs, [gamma_factor]*num_imgs,
					 [contrast_factor]*num_imgs, [sharpness_factor]*num_imgs, [color_factor]*num_imgs,
					 [norm_factor]*num_imgs)


def load_config_from_yaml(file_path):
	"""Load and return configuration parameters from a YAML file."""

	with open(file_path, 'r') as file:

		return yaml.safe_load(file)


def parse_args():
	"""Parse command-line arguments for synthetic dataset generation."""

	parser = argparse.ArgumentParser(description='Generate a synthetic dataset with templates.')
	
	parser.add_argument("--config", dest='config_path', type=str, required=True,
					 	help="Path to YAML configuration file.")
	parser.add_argument('--num-imgs', dest='num_imgs', type=int, required=True,
                        help='Number of images to be generated.')
	parser.add_argument('--max-process', dest='max_process', type=int, default=1,
                        help='Maximum number of parallel processes.')
	parser.add_argument('--debug', action='store_true', help='Run in debug mode to show bounding boxes on images.')

	args = parser.parse_args()
	return args


if __name__ == "__main__":

	args = parse_args()
	config_path = args.config_path
	num_imgs = args.num_imgs
	max_process = args.max_process
	debug = args.debug

	config = load_config_from_yaml(config_path)
	bgns_path = config['bgns_path']
	tpls_path = config['tpls_path']
	out_path = config['out_path']
	noise_bg = config['noise_bg']
	img_res = tuple(config['img_res'])
	orig_tpl_res = tuple(config['orig_tpl_res'])
	rescale_factor = config['rescale_factor']
	rotation_angle = config['rotation_angle']
	h_flip = config['h_flip']
	v_flip = config['v_flip']
	templ_number = config['templ_number']
	gamma_factor = config['gamma_factor']
	contrast_factor = config['contrast_factor']
	sharpness_factor = config['sharpness_factor']
	color_factor = config['color_factor']

	norm_factor = st.normalization_factor(orig_tpl_res, img_res)
	
	generate_dataset(bgns_path, tpls_path, out_path, num_imgs, max_process, img_res,
				  	 debug, noise_bg, rescale_factor, rotation_angle, h_flip, v_flip,
					 templ_number, gamma_factor, contrast_factor, sharpness_factor, 
					 color_factor, norm_factor)
