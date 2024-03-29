import os
import setuptools
openslide_path = r'C:\Python files\Software\openslide-win64-20231011\bin'
vipshome = r'C:\Python files\Software\vips-dev-w64-web-8.15.0\vips-dev-8.15\bin'
os.environ['PATH'] = vipshome
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(openslide_path):
        import dlup

from dlup.data.dataset import TiledWsiDataset
from dlup.experimental_backends import openslide_backend
from dlup import SlideImage
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from dlup.background_deprecated import get_mask  # use ahcore later
from cv2 import circle, HoughCircles, HOUGH_GRADIENT, bitwise_and, FILLED
import slidescore
import re
import shutil
from scipy import ndimage
from skimage import measure, color
from skimage.morphology import dilation, erosion, square
from skimage.filters import try_all_threshold, threshold_mean
import warnings


token = ('eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJOYW1lIjoiSEtTX1NhcmFfMjMxMjA0XyBJUkJtMTlfMTg1IiwiSUQiOiI1MzkzIiwiVmVy'
         'c2lvbiI6IjEuMCIsIkNhbkNyZWF0ZVVwbG9hZEZvbGRlcnMiOiJGYWxzZSIsIkNhblVwbG9hZCI6IkZhbHNlIiwiQ2FuRG93bmxvYWRTbGlkZ'
         'XMiOiJUcnVlIiwiQ2FuRGVsZXRlU2xpZGVzIjoiRmFsc2UiLCJDYW5VcGxvYWRPbmx5SW5Gb2xkZXJzIjoiIiwiQ2FuUmVhZE9ubHlTdHVkaW'
         'VzIjoiSVJCbTE5XzE4NV9FUjtJUkJtMTlfMTg1X0hFX3RyaXBsZV9uZWc7SVJCbTE5XzE4NV9IRTtJUkJtMTlfMTg1X0hFUjI7SVJCbTE5XzE'
         '4NV9LaV82NztJUkJtMTlfMTg1X1BSOyIsIkNhbk1vZGlmeU9ubHlTdHVkaWVzIjoiIiwiQ2FuR2V0Q29uZmlnIjoiRmFsc2UiLCJDYW5HZXRQ'
         'aXhlbHMiOiJUcnVlIiwiQ2FuSGFuZGxlRG9tYWlucyI6IkZhbHNlIiwiQ2FuSGFuZGxlUGF0aHMiOiJGYWxzZSIsIkNhblVwbG9hZFNjb3Jlc'
         'yI6IlRydWUiLCJDYW5DcmVhdGVTdHVkaWVzIjoiRmFsc2UiLCJDYW5SZWltcG9ydFN0dWRpZXMiOiJGYWxzZSIsIkNhbkRlbGV0ZU93bmVkU3'
         'R1ZGllcyI6IkZhbHNlIiwiQ2FuR2V0U2NvcmVzIjoiRmFsc2UiLCJDYW5HZXRBbnlTY29yZXMiOiJUcnVlIiwiQ2FuSGFuZGxlU3R1ZGVudEF'
         'jY291bnRzIjoiRmFsc2UiLCJuYmYiOjE3MDE2ODQ5MzYsImV4cCI6MTczMzI2NjgwMCwiaWF0IjoxNzAxNjg0OTM2fQ.gnwDKsJBlqk1YYmwn'
         'KZwyLCe_GGk24tzHlwfZDItb54#expires:2024-12-04T00:00:00')
url = "https://slidescore.nki.nl/"
studyid = 382

client = slidescore.APIClient(url, token)
client.get_studies()

download_path = r'C:\Users\n.ultee\PycharmProjects\ER_ITH_v1.0\IRBm19_185_ER'
download_res = r'R:\Groups\GroupLinn\Nigel\IRBm19_185_ER'

# download files that have not been downloaded yet
count = 0
for f in client.get_images(studyid):
    count = count + 1
    image_name = f["name"]  # works?
    pattern = re.compile(image_name + r'.(zip|svs)')
    match_C = list(filter(pattern.match, os.listdir(download_path)))
    match_R = list(filter(pattern.match, os.listdir(download_res)))
    if match_C:  # if downloaded in C-drive, move to R drive
        file_name = ''.join(match_C)
        shutil.move(os.path.join(download_path, file_name), download_res)
        print(count, ": ", image_name, "was previously downloaded in C-drive. Moved to R-drive.")
    elif match_R:  # if in R-drive, skip
        print(count, ": ", image_name, "was previously downloaded in R-drive.")
    else:  # if not downloaded, do so
        print(count, ": ", 'downloading ' + f["name"] + ' in R-drive...', end='', flush=True)
        client.download_slide(studyid, f["id"], download_res)
        print('done')
print('DOWNLOADING DONE')

################
### Pipeline ###
################
def remove_small_objects(original_mask, metric='mean', min_area_threshold=None, std_scale=None, range_scale=None):
    """
    Remove small objects from a binary mask.

    Parameters:
    - original_mask (ndarray): Binary mask representing objects.
    - min_area_threshold (int): Minimum area threshold for retaining objects, mean area by default.

    Returns:
    - new_mask (ndarray): New mask with small objects removed.

    Example:
    >>> new_mask = remove_small_objects(original_mask, min_area_threshold)
    """
    # TO DO:
    # 1) metric: mean, median, manual
    # 2) std_scale, range_scale, threshold
    # 3) make three different functions? And one central fucntion.
    # Label connected components in the mask
    labeled_mask, num_labels = measure.label(original_mask, connectivity=2, return_num=True)

    # Measure properties of labeled regions
    regions = measure.regionprops(labeled_mask)

    # If is not specified, threshold = mean + std
    areas = [region.area for region in regions]
    if metric == 'manual':
        min_area_threshold = min_area_threshold
    elif metric == 'median':
        if range_scale == None:
            min_area_threshold = np.median(areas) - 0.1 * np.median(areas)
        elif range_scale != None:
            min_area_threshold = np.median(areas) - range_scale * np.median(areas)
    else:
        if std_scale == None:
            min_area_threshold = np.mean(areas) + np.std(areas)
        elif std_scale != None:
            min_area_threshold = np.mean(areas) + std_scale * np.std(areas)

    print("Small object filter: ", sum(1 for area in areas if area > min_area_threshold),
          "objects retained.")

    # Create a new mask and add objects to be retained
    new_mask = np.zeros_like(original_mask, dtype=np.bool_)
    for region in regions:
        if region.area >= min_area_threshold:
            new_mask[labeled_mask == region.label] = True

    return new_mask

def generate_mask(image, kernel_dilation=10, kernel_erosion=3):
    """
    Generate a binary mask to separate foreground from background (tissue detection).

    Parameters:
    - image (ndarray): Input color image.
    - kernel_dilation (int): Size of the square kernel for dilation operation.
    - kernel_erosion (int): Size of the square kernel for erosion operation.
    - min_area_threshold (int): Minimum area threshold for removing small objects in the mask.

    Returns:
    - ndarray: Binary mask indicating segmented regions.

    Example:
    >>> input_image = imread('path/to/your/image.jpg')
    >>> result_mask = generate_mask(input_image, kernel_dilation=8, kernel_erosion=2, min_area_threshold=1000)
    """
    # Convert RGB image to grayscale
    gray_image = color.rgb2gray(image)

    # Perform dilation and erosion operations
    dilated_img = dilation(gray_image, square(kernel_dilation))
    eroded_img = erosion(gray_image, square(kernel_erosion))
    dilated_min_eroded_img = dilated_img - eroded_img

    # find good filter for segmentation
    # fig, ax = try_all_threshold(dilated_min_eroded_img, figsize=(10, 8), verbose=False)
    # plt.show()

    # Thresholding to create a binary mask
    threshold = threshold_mean(dilated_min_eroded_img)  # set threshold to segment (yes/no)
    inverse_binary = dilated_min_eroded_img <= threshold  # dark pixels==True, bright pixels==False
    mask = 1-inverse_binary

    # Remove small objects from the binary mask
    mask = remove_small_objects(mask, std_scale=0.5)

    return mask

def tile_image(file_path, TARGET_MPP, TILE_SIZE, mask):
    """
    Tile an image based on a given mask.

    Parameters:
    - file_path (str): Path to the whole slide image (WSI) file.
    - TARGET_MPP (float): Target microns per pixel (mpp) for scaling.
    - TILE_SIZE (tuple): Size of the tiles to be generated (width, height).
    - mask (ndarray): Binary mask indicating regions of interest.

    Returns:
    - ndarray: Tiled image with annotated regions.

    Example:
    >>> file_path = 'path/to/your/wsi_file.svs'
    >>> TARGET_MPP = 0.5
    >>> TILE_SIZE = (256, 256)
    >>> mask = np.array([[True, False, True], [False, True, False], [True, False, True]])
    >>> result_image = tile_image(file_path, TARGET_MPP, TILE_SIZE, mask)
    """
    # Tile image using mask
    scaled_region_view = slide_image.get_scaled_view(slide_image.get_scaling(TARGET_MPP))

    dataset = TiledWsiDataset.from_standard_tiling(path=file_path, mpp=TARGET_MPP, tile_size=TILE_SIZE,
                                                   tile_overlap=(0, 0), mask=mask,
                                                   backend=openslide_backend.OpenSlideSlide)
    tiled_image = Image.new("RGBA", tuple(scaled_region_view.size), (255, 255, 255, 255))  # instantiate new image
    for d in dataset:
        tile = d["image"]  # take tile (with tissues)
        coords = np.array(d["coordinates"])
        box = tuple(np.array((*coords, *(coords + TILE_SIZE))).astype(int))
        tiled_image.paste(tile, box)
        draw = ImageDraw.Draw(tiled_image)
        draw.rectangle(box, outline="red")


    return tiled_image

def retrieve_objects_count_per_side(regions, image_width):
    """
    Retrieve the count of objects on the left and right side of the slide.

    Parameters:
    - regions (list): List of region properties.
    - image_width (int): Width of the image.

    Returns:
    - tuple: A tuple containing two integers representing the count of objects on the left and right sides.
    """
    left_objects_count = 0
    right_objects_count = 0
    dividing_line = image_width // 2

    for region in regions:
        # Classify based on position
        if region.centroid[1] < dividing_line:
            left_objects_count += 1
        else:
            right_objects_count += 1

    return left_objects_count, right_objects_count

def retrieve_circularities_per_side(regions, image_width):
    """
    Retrieve circularity on left and right side of slide.

    Parameters:
    - regions (list): List of region properties.
    - image_width (int): Width of the image.

    Returns:
    - tuple: A tuple containing two lists of circularities for left and right objects.
    """
    left_circularities = []
    right_circularities = []
    dividing_line = image_width // 2

    for region in regions:
        # Calculate circularity
        circularity = 4 * np.pi * region.area / region.perimeter ** 2

        # Classify based on position
        if region.centroid[1] < dividing_line:
            left_circularities.append(circularity)
        else:
            right_circularities.append(circularity)

    return left_circularities, right_circularities

def max_circularity_method(mask, regions):
    """
    Add docstring
    """
    # Determine whether controls are located left or right
    width = mask.shape[1]
    height = mask.shape[0]
    left_circularities, right_circularities = retrieve_circularities_per_side(regions, image_width=width)

    # Decide which side to include based on mean circularity
    max_circularity_left = max(left_circularities)
    max_circularity_right = max(right_circularities)

    controls = np.zeros_like(mask)
    if max_circularity_left > max_circularity_right:
        # Include objects on the left side
        controls[0:height, 0:width // 2] = mask[0:height, 0:width // 2]
    elif max_circularity_left < max_circularity_right:
        # Include objects on the right side
        controls[0:height, width // 2:width] = mask[0:height, width // 2:width]
    else:
        # Warn if mean circularities are equal return original mask
        warnings.warn("Mean circularities on both sides are equal. Consider verifying the result.", UserWarning)

    return controls

    return mask

def median_circularity_method(mask, regions):
    """
    Add docstring
    """
    # Determine whether controls are located left or right
    width = mask.shape[1]
    height = mask.shape[0]
    left_circularities, right_circularities = retrieve_circularities_per_side(regions, image_width=width)

    # Decide which side to include based on mean circularity
    median_circularity_left = np.median(left_circularities)
    median_circularity_right = np.median(right_circularities)

    controls = np.zeros_like(mask)
    if median_circularity_left > median_circularity_right:
        # Include objects on the left side
        controls[0:height, 0:width // 2] = mask[0:height, 0:width // 2]
    elif median_circularity_left < median_circularity_right:
        # Include objects on the right side
        controls[0:height, width // 2:width] = mask[0:height, width // 2:width]
    else:
        # Warn if mean circularities are equal return original mask
        warnings.warn("Mean circularities on both sides are equal. Consider verifying the result.", UserWarning)

    return controls

def extract_round_objects_improved(mask,
                                   circularity_threshold=0.05,
                                   min_area_threshold=None,
                                   std_scale=None):
    """
    Extract round objects from a binary mask based on circularity.

    Parameters:
    - mask (ndarray): Binary mask indicating regions of interest.
    - circularity_threshold (float): Threshold for circularity to filter round objects, default is 0.1.

    Returns:
    - tuple: A tuple containing the following elements:
        - ndarray: Binary mask containing only the round objects.
        - list: List of circularities for all labeled regions.
        - list: List of circularities for included round objects.

    Example:
    >>> input_mask = np.array([[True, False, True], [False, True, False], [True, False, True]])
    >>> result_objects, circularities_all, circularities_included = extract_round_objects(input_mask, circularity_threshold=0.2)
    """
    # Second small objects filter (permitted here because only controls are needed)
    # TO DO: make this optional
    temp_mask = remove_small_objects(mask, metric='median', range_scale=0.2)

    # Label connected components in the binary mask
    labeled_mask, num_labels = measure.label(temp_mask, connectivity=2, return_num=True)

    # Measure properties of labeled regions
    regions = measure.regionprops(labeled_mask, intensity_image=temp_mask)

    width = mask.shape[1]
    objects_left, objects_right = retrieve_objects_count_per_side(regions, width)
    # if >1 object on >=1 sides do left_right, elif 0 objects on 1 side return original mask, else do max circularity
    if objects_left == 0 or objects_right == 0:
        warnings.warn("Zero objects detected on left/right side. Original mask will be returned", UserWarning)
        return mask
    elif objects_left == 1 or objects_right == 1:
        controls = max_circularity_method(mask, regions)
        return controls
    else:
        controls = median_circularity_method(mask, regions)
        return controls

# INPUT_PATH = r'C:\Users\n.ultee\PycharmProjects\ER_ITH_v1.0\IRBm19_185_ER\svs_files'
INPUT_PATH = r'C:\Users\n.ultee\PycharmProjects\ER_ITH_v1.0\IRBm19_185_ER\mrxs_files'
import zipfile

####
images = []
# tiled_images = []
slide_images = []
masks = []
regionprops_controls = []
# Iterate over each file in the folder
for filename in os.listdir(INPUT_PATH):
    FILE_PATH = os.path.join(INPUT_PATH, filename, (filename + '.mrxs'))

    # Check if it's a regular file
    if os.path.isfile(FILE_PATH):
        print(FILE_PATH)

    # Load image
    slide_image = SlideImage.from_file_path(FILE_PATH)

    # Convert image to numpy array
    max_slide = max(slide_image.size)
    size = max_slide * slide_image.mpp / 10  # Size is 10 mpp
    size = int(max([int(1 if int(size) == 0 else 2 ** (int(size) - 1).bit_length()), 512]))
    image = np.asarray(slide_image.get_thumbnail(size=(size, size)))

    # Rotate mrxs images
    image = np.rot90(image)

    images.append(image)  # remove later

    # Generate mask
    mask = generate_mask(image)
    masks.append(mask)

    # # Tile image
    # TARGET_MPP = 100  # microns per pixel
    # TILE_SIZE = (10, 10)  # does not cause bg detection problem
    # tiled_image = tile_image(FILE_PATH, TARGET_MPP, TILE_SIZE, mask)
    # tiled_images.append(tiled_image)  # remove later

    # Detect controls by round object detection (regionprops)
    round_objects = extract_round_objects_improved(mask, circularity_threshold=0.01)
    regionprops_controls.append(round_objects)

# Plot original images
num_rows = 4
num_cols = 2
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
for i in range(len(images)):
    row = i // num_cols
    col = i % num_cols
    axes[row, col].imshow(images[i], cmap='gray')
    axes[row, col].axis('off')  # Turn off axis labels
plt.show()

# Plot tiled images
num_rows = 4
num_cols = 2
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
for i in range(len(tiled_images)):
    row = i // num_cols
    col = i % num_cols
    axes[row, col].imshow(tiled_images[i], cmap='gray')
    axes[row, col].set_xticks([])  # Turn off x-axis ticks
    axes[row, col].set_yticks([])  # Turn off y-axis ticks
plt.show()

# Plot masks
num_rows = 4
num_cols = 2
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
for i in range(len(masks)):
    row = i // num_cols
    col = i % num_cols
    axes[row, col].imshow(masks[i], cmap='gray')
    axes[row, col].set_xticks([])  # Turn off x-axis ticks
    axes[row, col].set_yticks([])  # Turn off y-axis ticks
plt.show()

# Plot control images --> circularity
num_rows = 4
num_cols = 2
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
for i in range(len(regionprops_controls)):
    row = i // num_cols
    col = i % num_cols
    axes[row, col].imshow(regionprops_controls[i], cmap='gray')
    axes[row, col].set_xticks([])  # Turn off x-axis ticks
    axes[row, col].set_yticks([])  # Turn off y-axis ticks
plt.show()

########################################################################################################################
################################################## UNDER CONSTRUCTION ##################################################
########################################################################################################################

################
### Stardist ###
################
from stardist.models import StarDist2D
from stardist import fill_label_holes, random_label_cmap, render_label
from stardist.data import test_image_nuclei_2d
from csbdeep.utils import normalize

### Sreeni
img = test_image_nuclei_2d()  # uint16, needs normalization

# Load or train a Stardist model
model = StarDist2D.from_pretrained('2D_paper_dsb2018')

# Predict cell probabilities and labels
labels, _ = model.predict_instances(normalize(img))

# Visualize the results
cmap = random_label_cmap()
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(labels, cmap=cmap)
axes[1].set_title('Segmentation Labels')
plt.show()

### Import slide at specific level
import openslide
slide = openslide.OpenSlide(os.path.join(INPUT_PATH, filename))

# Get information about the levels
levels = slide.level_count
print(f"Number of levels: {levels}")

# Choose the level for analysis (e.g., level 1)
chosen_level = 0

# Read the image at the chosen level
# image_at_level = np.array(slide.read_region((0, 0), chosen_level, slide.level_dimensions[chosen_level]))[:, :, :3]
# plt.imshow(image_at_level)
# plt.show()

# Specify the region of interest (ROI) coordinates
roi_x, roi_y = 6000, 20000  # Example coordinates
roi_width, roi_height = 100, 100  # Example dimensions

# Read the image in the specified ROI
image_roi = np.array(slide.read_region((roi_x, roi_y), 0, (roi_width, roi_height)))[:, :, :3]
plt.imshow(image_roi)
plt.show()

# Close the slide object to free resources
slide.close()

# Convert the RGB image to grayscale if needed
sd_image = normalize(image_roi.astype(np.float32))

gray_image = 1 - np.mean(sd_image, axis=-1)
plt.imshow(gray_image, cmap="gray")
plt.show()

### Enhance edges ###
from skimage.filters import sobel
from scipy.ndimage import gaussian_filter

gradient_magnitude = sobel(sd_image)
sigma = gradient_magnitude.max()
smoothed_image = gaussian_filter(sd_image, sigma)
sobel_image = 1 - np.mean(smoothed_image, axis=-1)

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap="gray")
plt.axis("off")
plt.title("input image")

plt.subplot(1, 2, 2)
plt.imshow(sobel_image, cmap='gray')
plt.axis("off")
plt.title("sobel")

plt.show()
# it seems that the edges are even less well preserved
#################################

### Feature enhancement
from skimage.exposure import equalize_hist
enhanced_image = sd_image * gradient_magnitude
# enhanced_image = np.power(enhanced_image, 2)
enhanced_image = equalize_hist(enhanced_image)
inv_enhanced_image = 1 - np.mean(enhanced_image, axis=-1)

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap="gray")
plt.axis("off")
plt.title("Input image")

plt.subplot(1, 2, 2)
plt.imshow(inv_enhanced_image, cmap='gray')
plt.axis("off")
plt.title("Feature enhancement")

plt.show()
# only clear edges seem to become more clear, which does not improve cell segmentation
##############################

# Load the pre-trained Stardist model
model = StarDist2D.from_pretrained('2D_paper_dsb2018')

# Predict probabilities and labels
# labels, probabilities = model.predict(gray_image)  # label -> radial distance; probability -> pixel/point is center
labels, _ = model.predict_instances(normalize(inv_enhanced_image))

plt.subplot(1, 2, 1)
plt.imshow(inv_enhanced_image, cmap="gray")
plt.axis("off")
plt.title("input image")

plt.subplot(1, 2, 2)
plt.imshow(render_label(labels, img=inv_enhanced_image))
plt.axis("off")
plt.title("prediction + input overlay")

plt.show()

################
### Cellpose ###
################
# from cellpose import models

################ GRAVEYARD ################

def extract_round_objects(mask, circularity_threshold=0.1):
    """
    Extract round objects from a binary mask based on circularity.

    Parameters:
    - mask (ndarray): Binary mask indicating regions of interest.
    - circularity_threshold (float): Threshold for circularity to filter round objects, default is 0.1.

    Returns:
    - tuple: A tuple containing the following elements:
        - ndarray: Binary mask containing only the round objects.
        - list: List of circularities for all labeled regions.
        - list: List of circularities for included round objects.

    Example:
    >>> input_mask = np.array([[True, False, True], [False, True, False], [True, False, True]])
    >>> result_objects, circularities_all, circularities_included = extract_round_objects(input_mask, circularity_threshold=0.2)
    """
    # Label connected components in the binary mask
    labeled_image, num_labels = ndimage.label(mask)

    # Measure properties of labeled regions
    regions = measure.regionprops(labeled_image, intensity_image=mask)

    # Detect round object
    round_objects = []
    circularities = []  # for manually adjusting threshold
    circularities_included_object = []  # for manually adjusting threshold
    for region in regions:
        # filter on object area?
        circularity = 4 * np.pi * region.area / region.perimeter**2
        circularities.append(circularity)
        if circularity > circularity_threshold:
            round_objects.append(region)
            circularities_included_object.append(circularity)

    # Extract round objects from the original image
    extracted_objects = np.zeros_like(mask)
    for region in round_objects:
        extracted_objects[region.coords[:, 0], region.coords[:, 1]] = mask[region.coords[:, 0], region.coords[:, 1]]

    return extracted_objects, circularities, circularities_included_object


def extract_round_objects_improved(mask,
                                   circularity_threshold=0.05,
                                   min_area_threshold=None,
                                   std_scale=None):
    """
    Extract round objects from a binary mask based on circularity.

    Parameters:
    - mask (ndarray): Binary mask indicating regions of interest.
    - circularity_threshold (float): Threshold for circularity to filter round objects, default is 0.1.

    Returns:
    - tuple: A tuple containing the following elements:
        - ndarray: Binary mask containing only the round objects.
        - list: List of circularities for all labeled regions.
        - list: List of circularities for included round objects.

    Example:
    >>> input_mask = np.array([[True, False, True], [False, True, False], [True, False, True]])
    >>> result_objects, circularities_all, circularities_included = extract_round_objects(input_mask, circularity_threshold=0.2)
    """
    # Approach 1:
    # 1) Select most circular object
    # 2) Identify the location of this object
    # 3) Determine whether this is on the left or right side of the slide
    # 4) In mask, retain all surrounding objects, filter out objects on the other side
    #    ;we assume one of the controls will be the most circular ipv larger tissues
    # 5) Filter on radius (mean + x * std of left over objects OR %difference from radius most circular object)
    #    ;only if possible, otherwise just take all objects there.
    # In this approach, sometimes a small biopsy object has largest circularity. Extra small object filter might not
    # fully solve problem.

    # # Label connected components in the binary mask
    # #labeled_image, num_labels = ndimage.label(mask)
    # labeled_mask, num_labels = measure.label(mask, connectivity=2, return_num=True)
    #
    # # Measure properties of labeled regions
    # regions = measure.regionprops(labeled_mask, intensity_image=mask)

    # # Extra small objects filter
    # areas = [region.area for region in regions]
    # if min_area_threshold == None:
    #     if std_scale == None:
    #         min_area_threshold = np.mean(areas) + np.std(areas)
    #     elif std_scale != None:
    #         min_area_threshold = np.mean(areas) + std_scale * np.std(areas)

    # # Detect round objects
    # round_objects = []
    # circularities = []  # for manually adjusting threshold
    # max_circularity = 0.0
    # for region in regions:
    #     # Calculate circularity
    #     circularity = 4 * np.pi * region.area / region.perimeter ** 2
    #     circularities.append(circularity)
    #
    #     # 1) Select most circular object
    #     if circularity > max_circularity:
    #         max_circularity = circularity
    #         round_objects = [region]
    #         circularities_included_object = [circularity]
    #     elif circularity == max_circularity:
    #         # If circularity is the same, add the object to the list
    #         round_objects.append(region)
    #         circularities_included_object.append(circularity)


    # Approach 2:
    # 1) Divide slide
    # 2) Calculate mean circularity per side
    # 3) Retain objects in side with largest mean circularity
    # 4) Optionally: filter on radius.

    # height = int(mask.shape[0])
    # width = int(mask.shape[1])
    # half = int(width/2)
    # mask_left = mask[0:height, 0:half]
    # mask_right = mask[0:height, half:width]
    #
    # # Left side
    # labeled_mask_left, _ = measure.label(mask_left, connectivity=2, return_num=True)
    # regions_left = measure.regionprops(labeled_mask_left, intensity_image=mask_left)
    #
    # # Right side
    # labeled_mask_right, _ = measure.label(mask, connectivity=2, return_num=True)
    # regions_right = measure.regionprops(mask_right, intensity_image=mask_right)

    # Second small objects filter (permitted here because only controls are needed)
    # TO DO: make this optional
    mask = remove_small_objects(mask, metric='median', range_scale=0.1)

    # Label connected components in the binary mask
    labeled_mask, num_labels = measure.label(mask, connectivity=2, return_num=True)

    # Measure properties of labeled regions
    regions = measure.regionprops(labeled_mask, intensity_image=mask)

    # Determine whether controls are located left or right
    width = mask.shape[1]
    left_circularities, right_circularities = retrieve_circularities_per_side(regions, image_width=width)

    # Decide which side to include based on mean circularity
    mean_circularity_left = np.median(left_circularities)
    mean_circularity_right = np.median(right_circularities)

    if mean_circularity_left > mean_circularity_right:
        # Include objects on the left side
        mask_left = np.zeros_like(mask)
        for region in regions:
            if region.centroid[1] < width // 2:
                mask_left[region.coords[:, 0], region.coords[:, 1]] = mask[region.coords[:, 0], region.coords[:, 1]]
        return mask_left
    elif mean_circularity_left < mean_circularity_right:
        # Include objects on the right side
        mask_right = np.zeros_like(mask)
        for region in regions:
            if region.centroid[1] >= width // 2:
                mask_right[region.coords[:, 0], region.coords[:, 1]] = mask[region.coords[:, 0], region.coords[:, 1]]
        return mask_right
    else:
        # Warn if mean circularities are equal
        warnings.warn("Mean circularities on both sides are equal. Consider verifying the result.", UserWarning)
        return mask  # Return the original mask


def detect_circles(image, mask):
    """
    Detect and mark circles in an image.

    Parameters:
    - image (ndarray): The original image.
    - mask (ndarray): Binary mask representing areas of interest.

    Returns:
    - tuple: A tuple containing the following elements:
        - ndarray: The original image with detected circles drawn.
        - ndarray: Binary mask with circles drawn.
        - ndarray: Result of bitwise AND between the original mask and the mask with circles.

    Example:
    >>> image, control_mask, control_plus_biopsy = detect_circles(original_image, binary_mask)
    """
    mask_uint8 = mask.astype(np.uint8) * 255

    # Either mask of converting to uint8 results in edges that are too clear
    # Somehow filter out some pixels?
    # Extract edges binary mask
    gx, gy = np.gradient(1 - mask_uint8)
    mask_edges = gy * gy + gx * gx
    mask_edges[mask_edges != 0.0] = 255.0
    mask_edges = np.asarray(mask_edges, dtype=np.uint8)
    plt.imshow(mask_edges, cmap='gray')
    plt.show()

    # Apply Hough transform on the image
    circles = HoughCircles(mask_edges, HOUGH_GRADIENT, 1, image.shape[0] / 64, param1=200, param2=10, minRadius=120,
                                maxRadius=170)

    # Draw detected circles on the original image
    new_image = image.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw outer circle
            circle(new_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw inner circle
            circle(new_image, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Create a mask to store the areas of the detected circles
    control_mask = np.zeros_like(mask, dtype=np.uint8)

    # Draw detected circles on the mask
    if circles is not None:
        circles = np.uint8(np.around(circles))
        for i in circles[0, :]:
            # Draw circle on the mask
            circle(control_mask, (i[0], i[1]), i[2], 255, thickness=FILLED)

    # Apply the mask to the original mask
    control_plus_biopsy = bitwise_and(mask_uint8, control_mask)

    return new_image, control_mask, control_plus_biopsy


import numpy as np
from scipy.stats import hmean

def weighted_harmonic_mean(values):
    """
    Calculate the weighted harmonic mean of a list of values.

    Parameters:
    - values (list): List of values.

    Returns:
    - weighted_harmonic_mean (float): Weighted harmonic mean.
    """
    unique_values, frequencies = np.unique(values, return_counts=True)
    weights = 1 / unique_values
    weighted_harmonic_mean = hmean(unique_values, weights=frequencies)

    return weighted_harmonic_mean

def get_top_values(values, n=10):
    """
    Get the top n largest values from a list.

    Parameters:
    - values (list): List of values.
    - n (int): Number of top values to retrieve (default is 10).

    Returns:
    - top_values (list): List of the top n largest values.
    """
    top_values = np.sort(values)[-n:][::-1]
    return top_values

####
images = []
# tiled_images = []
slide_images = []
masks = []
regionprops_controls = []
# Iterate over each file in the folder
for filename in os.listdir(INPUT_PATH):
    FILE_PATH = os.path.join(INPUT_PATH, filename)

    # Check if it's a regular file
    if os.path.isfile(FILE_PATH):
        print(FILE_PATH)

    # Load image
    slide_image = SlideImage.from_file_path(FILE_PATH)

    # Convert image to numpy array
    max_slide = max(slide_image.size)
    size = max_slide * slide_image.mpp / 10  # Size is 10 mpp
    size = int(max([int(1 if int(size) == 0 else 2 ** (int(size) - 1).bit_length()), 512]))
    image = np.asarray(slide_image.get_thumbnail(size=(size, size)))
    images.append(image)  # remove later

    # Generate mask
    mask = generate_mask(image)
    masks.append(mask)

    # # Tile image
    # TARGET_MPP = 100  # microns per pixel
    # TILE_SIZE = (10, 10)  # does not cause bg detection problem
    # tiled_image = tile_image(FILE_PATH, TARGET_MPP, TILE_SIZE, mask)
    # tiled_images.append(tiled_image)  # remove later

    # Detect controls by round object detection (regionprops)
    round_objects = extract_round_objects_improved(mask, circularity_threshold=0.01)
    regionprops_controls.append(round_objects)