"""Image manipulation utilities for synthetic dataset generation.

Provides functions for template augmentation (scaling, rotation, flipping, filtering),
background generation, YOLO annotation formatting, and image-level color transforms.
"""

import os
import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from imgaug import augmenters as iaa


def set_yolo_annot_params(x1, y1, x2, y2, img_width, img_height, class_id):
    """Convert corner-format bounding box to normalized YOLO annotation format.

    Takes (x1, y1, x2, y2) corner coordinates and normalizes them to
    [class_id, x_center, y_center, width, height] with values in [0, 1].
    """

    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0

    width = x2 - x1
    height = y2 - y1

    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    return [class_id, x_center_norm, y_center_norm, width_norm, height_norm]


def calculate_bounding_box(position, template):
    """Calculate corner-format bounding box (x1, y1, x2, y2) from a paste position and template size."""

    return (
        position[0],  # top-left x
        position[1],  # top-left y
        position[0] + template.width,  # bottom-right x
        position[1] + template.height  # bottom-right y
    )


def overlay_template(background, template, position):
    """Paste an RGBA template onto the background at the given (x, y) position, respecting transparency."""
    background.paste(template, position, template)


def get_random_position(background, template):
    """Calculate a random valid position to place the template within the background bounds.

    Returns (x, y) tuple or None if the template is larger than the background.
    """

    max_x = background.width - template.width
    max_y = background.height - template.height
    if max_x < 0 or max_y < 0:
        return None  # Indicates the template is too large to be placed on the background
	
    return (random.randint(0, max_x), random.randint(0, max_y))


def color_transform(img, color_factor):
	"""Randomly adjust color saturation by a factor in the range [1 - factor, 1 + factor]."""
	enhancer = ImageEnhance.Color(img)
	img = enhancer.enhance(random.uniform((1.0 - color_factor), (1.0 + color_factor)))
	
	return img


def sharpness_transform(img, sharpness_factor):
	"""Randomly adjust sharpness by a factor in the range [1 - factor, 1 + factor]."""
	enhancer = ImageEnhance.Sharpness(img)
	img = enhancer.enhance(random.uniform((1.0 - sharpness_factor), (1.0 + sharpness_factor)))
	
	return img


def contrast_transform(img, contrast_factor):
	"""Randomly adjust contrast by a factor in the range [1 - factor, 1 + factor]."""

	enhancer = ImageEnhance.Contrast(img)
	img = enhancer.enhance(random.uniform((1.0 - contrast_factor), (1.0 + contrast_factor)))
	
	return img


def gamma_transform(img, gamma_factor):
	"""Randomly adjust brightness by a factor in the range [1 - factor, 1 + factor]."""

	enhancer = ImageEnhance.Brightness(img)
	img = enhancer.enhance(random.uniform((1.0 - gamma_factor), (1.0 + gamma_factor)))
	
	return img


def crop_transparent_borders(image):
	"""Crop an RGBA image to its non-transparent bounding box."""

	image = image.convert("RGBA")
	bbox = image.getbbox()
	if bbox:
		cropped_image = image.crop(bbox)
		return cropped_image
	
	return image

def rotate_img(template, rotation_angle):
    """Rotate the template by a random angle in [0, rotation_angle] and crop transparent borders."""

    angle = random.uniform(0, rotation_angle)
    new_template = template.rotate(angle, expand=True)
    new_template = crop_transparent_borders(new_template)

    return new_template


def flip_img(template, h_flip, v_flip):
    """Randomly flip the image horizontally (probability h_flip) and/or vertically (probability v_flip)."""
    if random.random() < h_flip:
        template = ImageOps.mirror(template)  # Horizontal flip
    if random.random() < v_flip:
        template = ImageOps.flip(template)  # Vertical flip
    return template


def rescale_img(img, rescale_factor):
	"""Randomly rescale the image by a factor in [1 - rescale_factor, 1 + rescale_factor]."""

	rescale_factor = random.uniform((1.0 - rescale_factor), (1.0 + rescale_factor))

	img_width, img_height = img.size

	new_width = int(img_width * rescale_factor)
	new_height = int(img_height * rescale_factor)
    
	img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

	return img


def normalize_img(img, norm_factor):
	"""Resize the template using a pre-computed normalization factor to match the target background resolution."""

	img_width, img_height = img.size

	new_width = int(img_width * norm_factor)
	new_height = int(img_height * norm_factor)
    
	img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

	return img


def normalization_factor(old_res, new_res):
	"""Compute the scaling factor to normalize templates from old_res to new_res, preserving aspect ratio."""

	old_width, old_height = old_res
	new_width, new_height = new_res
    
	scale_width = new_width/old_width
	scale_height = new_height/old_height
	
	return min(scale_width, scale_height)


def blending(image):
	"""Apply a multi-layer edge-blending filter to soften template borders.

    Creates 4 transparency layers (100%, 75%, 50%, 25%) with progressively
    eroded masks and composites them to produce a smooth transition at edges.
    """

	transparencies = [1.0, 0.75, 0.5, 0.25]
	images_with_transparency = []
    
	for transparency in transparencies:
		data = image.getdata() # (R, G, B, A) tuple list image data  
		new_data = []

		for item in data:
			new_alpha = int(item[3] * transparency)
			new_data.append((item[0], item[1], item[2], new_alpha))

		new_image = Image.new("RGBA", image.size)
		new_image.putdata(new_data)

		images_with_transparency.append(new_image)

	eroded_images = []
	kernel = np.ones((3, 3), np.uint8)

	for index, img in enumerate(images_with_transparency):
		img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
		eroded_img = img_cv

		for _ in range(3 - index):
			eroded_img = cv2.erode(eroded_img, kernel, iterations=1)

		eroded_images.append(Image.fromarray(cv2.cvtColor(eroded_img, cv2.COLOR_BGRA2RGBA)))

	base_image = eroded_images[3].copy()
    
	base_image = Image.alpha_composite(base_image, eroded_images[2])
	base_image = Image.alpha_composite(base_image, eroded_images[1])
	final_image = Image.alpha_composite(base_image, eroded_images[0])

	return final_image


def gaussian_blur(template):
	"""Apply a light Gaussian blur (sigma=0.6667) to reduce aliasing artifacts."""

	g_blur = iaa.GaussianBlur(sigma=0.6667)
	np_template = np.array(template)
	aug_template = g_blur.augment_image(np_template)
	aug_template = Image.fromarray(aug_template)

	return aug_template


def filter_template(template):
	"""Apply the full template filtering pipeline: Gaussian blur followed by edge blending."""
	
	template = gaussian_blur(template)
	template = blending(template)
	
	return template


def get_random_image(folder):
    """Load and return a random image from the given directory."""

    return Image.open(os.path.join(folder, random.choice(os.listdir(folder))))


def resize_img(img, img_resolution):
	"""Resize an image to the exact specified (width, height) resolution."""
	
	width, height = img_resolution
	img = img.resize((width, height), Image.Resampling.LANCZOS)
	
	return img

def noise_background(img_resolution):
	"""Generate a random RGB noise image with the given (width, height) resolution."""
   
	width, height = img_resolution
    
	noise_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
	img_noise = Image.fromarray(noise_array)
    
	return img_noise