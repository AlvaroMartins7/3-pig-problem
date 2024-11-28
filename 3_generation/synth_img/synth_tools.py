import os
import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from imgaug import augmenters as iaa

# Return in YOLO format annotation
def set_yolo_annot_params(x1, y1, x2, y2, img_width, img_height, class_id):

    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0

    width = x2 - x1
    height = y2 - y1

    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    return [class_id, x_center_norm, y_center_norm, width_norm, height_norm]


# Function to calculate the bounding box coordinates of a template
def calculate_bounding_box(position, template):

    return (
        position[0],  # top-left x
        position[1],  # top-left y
        position[0] + template.width,  # bottom-right x
        position[1] + template.height  # bottom-right y
    )


# Function to overlay the template onto the background at the given position
def overlay_template(background, template, position):
    background.paste(template, position, template)


# Function to calculate a random position within the background for the template
def get_random_position(background, template):

    max_x = background.width - template.width
    max_y = background.height - template.height
    if max_x < 0 or max_y < 0:
        return None  # Indicates the template is too large to be placed on the background
	
    return (random.randint(0, max_x), random.randint(0, max_y))


# Increases or decreases the contrast of a given image using a factor
def color_transform(img, color_factor):

	enhancer = ImageEnhance.Color(img)
	img = enhancer.enhance(random.uniform((1.0 - color_factor), (1.0 + color_factor)))
	
	return img


# Increases or decreases the contrast of a given image using a factor
def sharpness_transform(img, sharpness_factor):

	enhancer = ImageEnhance.Sharpness(img)
	img = enhancer.enhance(random.uniform((1.0 - sharpness_factor), (1.0 + sharpness_factor)))
	
	return img


# Increases or decreases the contrast of a given image using a factor
def contrast_transform(img, contrast_factor):

	enhancer = ImageEnhance.Contrast(img)
	img = enhancer.enhance(random.uniform((1.0 - contrast_factor), (1.0 + contrast_factor)))
	
	return img


# Increases or decreases the brightness of a given image using a factor
def gamma_transform(img, gamma_factor):

	enhancer = ImageEnhance.Brightness(img)
	img = enhancer.enhance(random.uniform((1.0 - gamma_factor), (1.0 + gamma_factor)))
	
	return img


#Crop the image to the non-transparent parts.
def crop_transparent_borders(image):

    image = image.convert("RGBA")
    bbox = image.getbbox()
    if bbox:
        cropped_image = image.crop(bbox)
        return cropped_image
	
    return image

# Function to rotate the template using a angle range
def rotate_img(template, rotation_angle):

    angle = random.uniform(0, rotation_angle)
    new_template = template.rotate(angle, expand=True)
    new_template = crop_transparent_borders(new_template)

    return new_template


# Function to randomly flip the image horizontally or vertically
def flip_img(template, h_flip, v_flip):
    if random.random() < h_flip:
        template = ImageOps.mirror(template)  # Horizontal flip
    if random.random() < v_flip:
        template = ImageOps.flip(template)  # Vertical flip
    return template


# Function to rescale a image using a scaling factor range
def rescale_img(img, rescale_factor):

	rescale_factor = random.uniform((1.0 - rescale_factor), (1.0 + rescale_factor))

	img_width, img_height = img.size

	new_width = int(img_width * rescale_factor)
	new_height = int(img_height * rescale_factor)
    
	img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

	return img


# Function to rescale a image using the background scaling factor
def normalize_img(img, norm_factor):

	img_width, img_height = img.size

	new_width = int(img_width * norm_factor)
	new_height = int(img_height * norm_factor)
    
	img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

	return img


# returns the scaling factor to normalize the templates for different background resolutions
def normalization_factor(old_res, new_res):

	old_width, old_height = old_res
	new_width, new_height = new_res
    
	scale_width = new_width/old_width
	scale_height = new_height/old_height
	
	return min(scale_width, scale_height)


# Applies a 4-step filter to a given image to help it blend to a background
def blending(image):

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


# Blurs a given image using gaussian distribution rule
def gaussian_blur(template):

	g_blur = iaa.GaussianBlur(sigma=0.6667)
	np_template = np.array(template)
	aug_template = g_blur.augment_image(np_template)
	aug_template = Image.fromarray(aug_template)

	return aug_template


# Applies Gaussian Blur, then a Blending filter
def filter_template(template):
	
	template = gaussian_blur(template)
	template = blending(template)
	
	return template


# Function to randomly choose an image from a folder
def get_random_image(folder):

    return Image.open(os.path.join(folder, random.choice(os.listdir(folder))))


# Resizes a image to a predefined resolution
def resize_img(img, img_resolution):
	
	width, height = img_resolution
	img = img.resize((width, height), Image.Resampling.LANCZOS)
	
	return img

# Generates a noise background with the given resolution
def noise_background(img_resolution):
   
	width, height = img_resolution
    
	noise_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
	img_noise = Image.fromarray(noise_array)
    
	return img_noise