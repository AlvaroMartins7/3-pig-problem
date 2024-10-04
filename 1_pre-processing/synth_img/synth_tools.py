import os
import random
from PIL import Image


def set_yolo_annot_params(x1, y1, x2, y2, img_width, img_height, class_id):
    # Calculate center of the bounding box
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0

    # Calculate width and height of the bounding box
    width = x2 - x1
    height = y2 - y1

    # Normalize values by the image size
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    # Return in YOLO format
    return [class_id, x_center_norm, y_center_norm, width_norm, height_norm]

# Function to randomly choose an image from a folder
def get_random_image(folder):
    return Image.open(os.path.join(folder, random.choice(os.listdir(folder))))

# Function to randomly resize the template by up to 50% of its original size
def random_resize_template(template):
    scale_factor = random.uniform(0.5, 1.0)
    new_width = int(template.width * scale_factor)
    new_height = int(template.height * scale_factor)
    resized_template = template.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_template

# Function to calculate a random position within the background for the template
def get_random_position(background, template):
    max_x = background.width - template.width
    max_y = background.height - template.height
    if max_x < 0 or max_y < 0:
        return None  # Indicates the template is too large to be placed on the background
    return (random.randint(0, max_x), random.randint(0, max_y))

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