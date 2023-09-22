import cv2
import numpy as np
import os


# Load the image and mask TIFF files
image_file =r"I:\Common\Uttam\Data\Electron microscopy (EM) dataset\images\training.tif"
mask_file = r"I:\Common\Uttam\Data\Electron microscopy (EM) dataset\masks\training_groundtruth.tif"
output_image_directory = r"I:\Common\Uttam\Data\Electron microscopy (EM) dataset\images\data\images"
output_mask_directory = r"I:\Common\Uttam\Data\Electron microscopy (EM) dataset\images\data\masks"
patch_size = (256, 256)

# Read the image and mask TIFF files
image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)

# Get the dimensions of the input image
height, width = image.shape

# Calculate the number of patches in both dimensions
num_patches_height = height // patch_size[0]
num_patches_width = width // patch_size[1]

# Create the output directories if they don't exist
os.makedirs(output_image_directory, exist_ok=True)
os.makedirs(output_mask_directory, exist_ok=True)

# Loop through the image and mask, and extract patches
for i in range(num_patches_height):
    for j in range(num_patches_width):
        # Calculate the coordinates for the top-left corner of the patch
        y_start = i * patch_size[0]
        x_start = j * patch_size[1]
        
        # Extract the image patch and mask patch
        image_patch = image[y_start:y_start + patch_size[0], x_start:x_start + patch_size[1]]
        mask_patch = mask[y_start:y_start + patch_size[0], x_start:x_start + patch_size[1]]
        
        # Save the image patch and mask patch in separate folders
        image_patch_filename = f"{output_image_directory}/image_patch_{i}_{j}.tif"
        mask_patch_filename = f"{output_mask_directory}/mask_patch_{i}_{j}.tif"
        
        cv2.imwrite(image_patch_filename, image_patch)
        cv2.imwrite(mask_patch_filename, mask_patch)

print(f"{num_patches_height * num_patches_width} image and mask patches generated and saved in separate folders.")

