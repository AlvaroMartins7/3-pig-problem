import os
import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from imgaug import augmenters as iaa

# Kernel global para erosão (reutilizado)
_EROSION_KERNEL = np.ones((3, 3), np.uint8)

# Cache para lista de arquivos em pastas para get_random_image
_image_folder_cache = {}


def preload_templates(template_folder):
    """
    Carrega todas as imagens da pasta template_folder em memória,
    retornando uma lista de objetos PIL.Image.
    """
    templates = []
    for filename in os.listdir(template_folder):
        full_path = os.path.join(template_folder, filename)
        if os.path.isfile(full_path):
            try:
                img = Image.open(full_path).convert('RGBA')
                templates.append(img)
            except Exception as e:
                print(f"Erro ao abrir template {full_path}: {e}")
    return templates

# Return in COCO format annotation
def set_coco_annot_params(x1, y1, x2, y2, image_id, category_id, annotation_id):
    width = x2 - x1
    height = y2 - y1
    area = width * height
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x1, y1, width, height],  # [top-left-x, top-left-y, width, height]
        "area": area,
        "iscrowd": 0
    }
    return annotation


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


# Increases or decreases the color saturation of a given image using a factor
def color_transform(img, color_factor):
    enhancer = ImageEnhance.Color(img)
    factor = random.uniform((1.0 - color_factor), (1.0 + color_factor))
    img = enhancer.enhance(factor)
    return img


# Increases or decreases the sharpness of a given image using a factor
def sharpness_transform(img, sharpness_factor):
    enhancer = ImageEnhance.Sharpness(img)
    factor = random.uniform((1.0 - sharpness_factor), (1.0 + sharpness_factor))
    img = enhancer.enhance(factor)
    return img


# Increases or decreases the contrast of a given image using a factor
def contrast_transform(img, contrast_factor):
    enhancer = ImageEnhance.Contrast(img)
    factor = random.uniform((1.0 - contrast_factor), (1.0 + contrast_factor))
    img = enhancer.enhance(factor)
    return img


# Increases or decreases the brightness of a given image using a factor
def gamma_transform(img, gamma_factor):
    enhancer = ImageEnhance.Brightness(img)
    factor = random.uniform((1.0 - gamma_factor), (1.0 + gamma_factor))
    img = enhancer.enhance(factor)
    return img


# Crop the image to the non-transparent parts.
def crop_transparent_borders(image):
    image = image.convert("RGBA")
    bbox = image.getbbox()
    if bbox:
        cropped_image = image.crop(bbox)
        return cropped_image
    return image


# Function to rotate the template using an angle range
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
    factor = random.uniform((1.0 - rescale_factor), (1.0 + rescale_factor))
    img_width, img_height = img.size
    new_width = int(img_width * factor)
    new_height = int(img_height * factor)
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img


# Function to rescale an image using the background scaling factor (fixed factor)
def normalize_img(img, norm_factor):
    img_width, img_height = img.size
    new_width = int(img_width * norm_factor)
    new_height = int(img_height * norm_factor)
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img


# Calculates the scaling factor to normalize the templates for different background resolutions
def normalization_factor(old_res, new_res):
    old_width, old_height = old_res
    new_width, new_height = new_res
    scale_width = new_width / old_width
    scale_height = new_height / old_height
    return min(scale_width, scale_height)


# Applies a 4-step filter to a given image to help it blend to a background
def blending(image, collect_steps=False):
    # image is a PIL RGBA image
    transparencies = [1.0, 0.75, 0.5, 0.25]

    np_img = np.array(image)  # shape (H, W, 4), dtype=uint8
    processed_images = []
    alpha_images = []
    erosion_images = []

    for idx, alpha_factor in enumerate(transparencies):
        # First erode the template border, then adjust transparency
        img_cv = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGRA)

        iterations = 3 - idx
        if iterations > 0:
            eroded_img = cv2.erode(img_cv, _EROSION_KERNEL, iterations=iterations)
        else:
            eroded_img = img_cv

        eroded_pil = Image.fromarray(cv2.cvtColor(eroded_img, cv2.COLOR_BGRA2RGBA))
        if collect_steps:
            erosion_images.append(eroded_pil.copy())

        alpha_adjusted = np.array(eroded_pil)
        alpha_adjusted[..., 3] = (alpha_adjusted[..., 3].astype(np.float32) * alpha_factor).astype(np.uint8)
        alpha_pil = Image.fromarray(alpha_adjusted, mode='RGBA')
        processed_images.append(alpha_pil)
        if collect_steps:
            alpha_images.append(alpha_pil.copy())

    # Composite images with alpha blending
    base_image = processed_images[3].copy()
    base_image = Image.alpha_composite(base_image, processed_images[2])
    base_image = Image.alpha_composite(base_image, processed_images[1])
    final_image = Image.alpha_composite(base_image, processed_images[0])

    if collect_steps:
        return final_image, alpha_images, erosion_images

    return final_image


# Blurs a given image using gaussian distribution rule
def gaussian_blur(template):
    g_blur = iaa.GaussianBlur(sigma=0.6667)
    np_template = np.array(template)
    aug_template = g_blur.augment_image(np_template)
    aug_template = Image.fromarray(aug_template)
    return aug_template


# Applies Gaussian Blur, then a Blending filter
def filter_template(template, collect_steps=False):
    template = gaussian_blur(template)
    if collect_steps:
        final_template, alpha_images, erosion_images = blending(template, collect_steps=True)
        return final_template, alpha_images, erosion_images

    template = blending(template)
    return template


# Function to randomly choose an image from a folder (with caching)
def get_random_image(folder):
    if folder not in _image_folder_cache:
        _image_folder_cache[folder] = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    file_list = _image_folder_cache[folder]
    if not file_list:
        raise ValueError(f"No files found in folder {folder}")
    filename = random.choice(file_list)
    return Image.open(os.path.join(folder, filename))


# Resizes an image to a predefined resolution
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
