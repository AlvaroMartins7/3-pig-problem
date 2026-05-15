import pandas as pd
from PIL import Image


def has_intersection(rect1, rect2):
    """Check whether two rectangles overlap using axis-aligned bounding box collision.

    Each rectangle dict must contain 'x', 'y', 'width', and 'height' keys,
    where (x, y) is the top-left corner.

    Returns True if the rectangles intersect, False otherwise.
    """
    x1_min, y1_min, x1_max, y1_max = rect1['x'], rect1['y'], rect1['x'] + rect1['width'], rect1['y'] + rect1['height']
    x2_min, y2_min, x2_max, y2_max = rect2['x'], rect2['y'], rect2['x'] + rect2['width'], rect2['y'] + rect2['height']
    
    return not (x1_max <= x2_min or x1_min >= x2_max or y1_max <= y2_min or y1_min >= y2_max)

# Load the CSV annotation file exported from VIA (VGG Image Annotator)
df = pd.read_csv('data/12-00/12-00.csv')

# Iterate over each unique image in the dataset
for img_filename in df['filename'].unique():
    # Filter annotations belonging to this specific image
    img_annotations = df[df['filename'] == img_filename]
    
    # Set to store row indices of regions involved in overlapping conflicts
    conflict_regions = set()

    # Compare every pair of annotations to detect bounding-box overlaps
    for i, row1 in img_annotations.iterrows():
        rect1 = eval(row1['region_shape_attributes'])  # Parse the JSON-like string into a dict
        current_rect1 = {
            'x': rect1['x'],
            'y': rect1['y'],
            'width': rect1['width'],
            'height': rect1['height']
        }

        # Compare against all other annotated regions in the same image
        for j, row2 in img_annotations.iterrows():
            if i != j:  # Skip self-comparison
                rect2 = eval(row2['region_shape_attributes'])
                current_rect2 = {
                    'x': rect2['x'],
                    'y': rect2['y'],
                    'width': rect2['width'],
                    'height': rect2['height']
                }
                
                if has_intersection(current_rect1, current_rect2):
                    # Mark both overlapping regions as conflicting
                    conflict_regions.add(i)
                    conflict_regions.add(j)

    # Extract and save only the non-overlapping (conflict-free) regions as templates
    for index, row in img_annotations.iterrows():
        if index not in conflict_regions:
            shape_attributes = eval(row['region_shape_attributes'])
            x = shape_attributes['x']
            y = shape_attributes['y']
            width = shape_attributes['width']
            height = shape_attributes['height']

            # Open the source image and crop the annotated region
            img_path = f'data/12-00/{img_filename}'
            with Image.open(img_path) as img:
                cropped_img = img.crop((x, y, x + width, y + height))
                
                # Save the cropped template to the output directory
                cropped_img.save(f'output/{img_filename}_region_{row["region_id"]}.jpg')

print("Extraction complete. All overlapping regions were skipped.")
