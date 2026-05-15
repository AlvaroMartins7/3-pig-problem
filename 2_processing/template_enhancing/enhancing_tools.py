"""Utility functions for SAM2-based template enhancement.

Provides mask application and content-aware cropping to produce clean RGBA
templates with transparent backgrounds from SAM2 segmentation masks.
"""

from PIL import Image
import numpy as np


def apply_mask_and_remove(mask, image):
    """Apply a binary segmentation mask to an image as an alpha channel.

    Pixels inside the mask (value=1) remain opaque; pixels outside the mask
    become fully transparent. Returns an RGBA numpy array.
    """
    # Convert image to RGBA (to add transparency)
    image = Image.fromarray(image).convert("RGBA")
    image = np.array(image)
    
    # Ensure the mask is a binary mask (0s and 1s)
    mask = mask.astype(np.uint8)
    
    # Create an alpha channel: 255 (opaque) for masked area, 0 (transparent) for background
    alpha_channel = np.where(mask == 1, 255, 0).astype(np.uint8)
    image[:, :, 3] = alpha_channel
    return image


def crop_to_content(image):
    """Crop an RGBA image to its non-transparent bounding box.

    Finds the tightest axis-aligned bounding box around all non-transparent
    pixels and returns the cropped region. If the image is fully transparent,
    returns it unchanged.
    """
    alpha_channel = image[:, :, 3]
    non_transparent_pixels = np.where(alpha_channel > 0)

    # Get the bounding box of the non-transparent area
    if non_transparent_pixels[0].size > 0 and non_transparent_pixels[1].size > 0:
        min_y = np.min(non_transparent_pixels[0])
        max_y = np.max(non_transparent_pixels[0])
        min_x = np.min(non_transparent_pixels[1])
        max_x = np.max(non_transparent_pixels[1])
        
        # Crop the image to this bounding box
        cropped_image = image[min_y:max_y+1, min_x:max_x+1, :]
        return cropped_image
    else:
        # If the whole image is transparent, return the original image
        return image