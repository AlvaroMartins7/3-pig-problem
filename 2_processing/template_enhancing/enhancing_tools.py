from PIL import Image
import numpy as np

def apply_mask_and_remove(mask, image):
    """
    Apply the mask to the image and remove the masked section by making it black.
    """
    # Convert image to RGBA (to add transparency)
    image = Image.fromarray(image).convert("RGBA")
    image = np.array(image)
    
    # Ensure the mask is a binary mask (0s and 1s)
    mask = mask.astype(np.uint8)
    
    # Create an alpha channel: 255 for the masked area, 0 for non-masked areas (transparent)
    alpha_channel = np.where(mask == 1, 255, 0).astype(np.uint8)
    # Apply the alpha channel to the image (set non-masked areas to transparent)
    image[:, :, 3] = alpha_channel
    return image

def crop_to_content(image):
    """
    Crop the image to the non-transparent parts.
    """
    # Find where the alpha channel is non-zero (i.e., non-transparent areas)
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