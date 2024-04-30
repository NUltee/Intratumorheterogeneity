import os
import setuptools
import zipfile
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
from openslide import OpenSlide
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from dlup.background_deprecated import get_mask  # use ahcore later
# from cv2 import circle, HoughCircles, HOUGH_GRADIENT, bitwise_and, FILLED
import slidescore
import re
import shutil
from scipy import ndimage
from skimage import measure, color
from skimage.morphology import dilation, erosion, square
from skimage.filters import try_all_threshold, threshold_mean, threshold_otsu
from skimage.exposure import equalize_adapthist
import warnings
from cv2 import findContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, boundingRect

#############################
### Download IRBdm 19-185 ###
#############################
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
results = client.get_results(studyid)
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

def find_id_by_name(images_info, t_number):
    """
    Find the 'id' corresponding to the specified t-number in a list of dictionaries.

    Parameters:
        data (list[dict]): List of dicts containing 'id' and 'name' entries, yielded by client.get_images(studyid)
        t_number (str): The T-number corresponsding to the slide.

    Returns:
        int or None: The 'id' corresponding to the specified 'name', or None if not found.
    """
    for item in images_info:
        if item.get('name') == t_number:
            return item.get('id')
    return None

def check_zip_files(folder_path, images_info):
    """
    Check all zip files in the specified folder and attempt to read each one.

    Parameters:
        folder_path (str): The path to the folder containing the zip files.

    Returns:
        None

    This function iterates over all zip files in the specified folder and attempts to open each zip file using the
    zipfile module. If a zip file can be successfully opened, it lists the contents of the zip file. If a zip file
    encounters an error during opening or processing, it removes the zip file from the folder and downloads the folder
    again.
    """
    # List all files in the specified folder
    files = os.listdir(folder_path)

    # Filter out only the zip files
    zip_files = [file for file in files if file.endswith('.zip')]

    # Iterate over each zip file
    for zip_file in zip_files:
        file_path = os.path.join(folder_path, zip_file)
        try:
            # Try to open the zip file
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Attempt to list contents (this will check if it's readable)
                pass
                file_list = zf.namelist()
                print(f"Zip file '{zip_file}' is readable and contains the following files:")
                # for file in file_list:
                #     print(f"  - {file}")
        except zipfile.BadZipFile:
            # This exception will be raised if the file is not a valid zip file
            print(f"Error: '{zip_file}' is not a valid zip file.")
            os.remove(file_path)

            t_number = zip_file[:-4] if zip_file.endswith('.zip') else zip_file
            slide_id = find_id_by_name(images_info, t_number)
            if slide_id is not None:
                client.download_slide(studyid, slide_id, download_res)
                print(f"Removed and redownloaded '{zip_file}'.")
        except Exception as e:
            # Catch any other exceptions
            print(f"Error: '{zip_file}' encountered an error - {e}")
            os.remove(file_path)

            t_number = zip_file[:-4] if zip_file.endswith('.zip') else zip_file
            slide_id = find_id_by_name(images_info, t_number)
            if slide_id is not None:
                client.download_slide(studyid, slide_id, download_res)
                print(f"Removed and redownloaded '{zip_file}'.")

# Call the function to check zip files in the specified folder
check_zip_files(download_res, client.get_images(studyid))
print('CHECKING ZIP FOLDERS DONE')

################
### Pipeline ###
################
def get_slide_bounds(slide):
    """
    Add docstring
    """
    bound_width = int(slide.properties.get('openslide.bounds-width'))
    bound_height = int(slide.properties.get('openslide.bounds-height'))
    bound_x = int(slide.properties.get('openslide.bounds-x'))
    bound_y = int(slide.properties.get('openslide.bounds-y'))

    return bound_width, bound_height, bound_x, bound_y

def crop_to_slide_bounds(slide, level_from, level_to):
    """
    Scale slide bounds from one level to another.

    Parameters:
    - slide: Openslide object
    - level_from (int): Source level
    - level_to (int): Target level

    Returns:
    - image_at_level (numpy array): Pixel values within slide bounds at specific image pyramidal level.
    """
    downsample_factor = slide.level_downsamples[level_from] / slide.level_downsamples[level_to]

    width, height, x, y = get_slide_bounds(slide)

    scaled_height = int(height * downsample_factor)
    scaled_width = int(width * downsample_factor)
    # scaled_x = int(x * downsample_factor)  # apparently you can just take the location as in level 0
    # scaled_y = int(y * downsample_factor)

    # Read the image at the chosen level
    image_at_level = np.array(slide.read_region((x, y), level_to, (scaled_width, scaled_height)))[:, :, :3]

    return image_at_level

def get_foreground_contour(image):
    """
    Retrieve the contour width and height of the full foreground (rectangular).

    Parameters:
    - binary_image (np array): binary image.

    Returns:
    - contour_width (int): width of rectangle contour around foreground.
    - contour_height (int): height of rectangle contour around foreground.
    """
    # Find contours in the binary image
    image = image.astype(np.uint8)
    contours, _ = findContours(image, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    # RETR_EXTERNAL -> retrieves only the external contours of the shapes (ignores shapes within other shapes).
    # CHAIN_APPROX_SIMPLE -> compresses horizontal, diagonal, and vertical segments and leaves only their end points.

    # Calculate the bounding box coordinates for the entire foreground
    top_left_x, top_left_y, contour_width, contour_height = boundingRect(np.vstack(contours))

    return (top_left_x, top_left_y), (contour_width, contour_height)

def measure_properties(binary_image):
    """
    Add docstring
    """
    # Measure properties of all gaps
    labeled_mask, num_labels = measure.label(binary_image, connectivity=2, return_num=True)  # label objects
    regions = measure.regionprops(labeled_mask, intensity_image=binary_image)  # measure properties

    return regions

def remove_small_objects(original_mask, metric='mean', min_area_threshold=None, std_scale=None, range_scale=None,
                         percentage_max_area=None):
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
    # Label connected components in the mask
    labeled_mask, num_labels = measure.label(original_mask, connectivity=2, return_num=True)

    # Measure properties of labeled regions
    regions = measure.regionprops(labeled_mask)

    # If not specified, threshold = mean + std
    # TO DO: determine threshold in other function.
    areas = [region.area for region in regions]
    if metric == 'manual':
        min_area_threshold = min_area_threshold
    elif metric == 'median':
        if range_scale == None:
            min_area_threshold = np.median(areas) - 0.1 * np.median(areas)
        elif range_scale != None:
            min_area_threshold = np.median(areas) - range_scale * np.median(areas)
    elif metric == 'maximum':
        if percentage_max_area == None:
            min_area_threshold = 0.01 * np.max(areas)
        elif percentage_max_area != None:
            min_area_threshold = percentage_max_area/100 * np.max(areas)
    else:
        if std_scale == None:
            min_area_threshold = np.mean(areas) + np.std(areas)
        elif std_scale != None:
            min_area_threshold = np.mean(areas) + std_scale * np.std(areas)

    # Create a new mask and add objects to be retained
    if len(areas) > 3:  # ensure more than 3 objects >threshold to filter
        print("Small object filter: ", sum(1 for area in areas if area > min_area_threshold),
              "objects retained.")
        new_mask = np.zeros_like(original_mask, dtype=np.bool_)
        for region in regions:
            if region.area >= min_area_threshold:
                new_mask[labeled_mask == region.label] = True
        return new_mask
    else:
        print("<3 Objects above threshold, original mask returned.")
        return original_mask

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
    mask = dilated_min_eroded_img >= threshold  # dark pixels==False, bright pixels==True

    # Remove small objects from the binary mask
    mask = remove_small_objects(mask, metric='maximum', percentage_max_area=2)

    return mask

def to_landscape(image):
    """
    Rotate binary image to landscape if orientation is portrait.

    Parameters:
    - image (np array): image.

    Returns:
    - landscape (np array): rotated binary image if foreground is not already in landscape orientation.
    - binary_image (np array): original binary image if foreground is already in landscape orientation.
    """
    # TO DO: rotate original image as well
    image = image.astype(np.uint8)
    _, dimensions = get_foreground_contour(image)
    width, height = dimensions
    if height > width:
        landscape = np.rot90(image)
        return landscape
    elif width >= height:
        return image

def controls_to_portrait(image, controls):
    """
    Rotate binary image to portrait if orientation is landscape.

    Parameters:
    - image (np array): image.

    Returns:
    - landscape (np array): rotated binary image if foreground is not already in landscape orientation.
    - binary_image (np array): original binary image if foreground is already in landscape orientation.
    """
    # TO DO: rotate original image as well
    image = image.astype(np.uint8)
    _, dimensions = get_foreground_contour(image)
    width, height = dimensions
    if height > width:
        return controls
    elif width >= height:
        portrait = np.rot90(controls, k=3)  # rotate back to original position
        return portrait

def crop_to_contour(mask):
    """
    Add docstring
    mask: binary mask
    """
    # Find location and dimensions of tissue to crop to
    location, dimensions = get_foreground_contour(mask)
    top_left_x, top_left_y = location
    width, height = dimensions
    mask = mask[top_left_y:top_left_y + height, top_left_x:top_left_x + width]

    return mask

def find_gaps_horizontally(mask, threshold=None):
    """
    Add docstring
    """
    # Turn all values in column to 1 if there are at least as many 1's equal or above the threshold
    if threshold == None:
        threshold = 5
    columns_with_ones = np.sum(mask, axis=0) >= threshold
    temp_mask = np.zeros_like(mask)
    temp_mask[:, columns_with_ones] = 1
    gaps = ~temp_mask  # invert to find gaps

    return gaps

def find_gaps_vertically(mask, threshold=None):
    """
    Add docstring
    """
    # Turn all values in column to 1 if there are at least as many 1's equal or above the threshold
    if threshold == None:
        threshold = 5
    rows_with_ones = np.sum(mask, axis=1) >= threshold
    temp_mask = np.zeros_like(mask)
    temp_mask[rows_with_ones, :] = 1
    gaps = ~temp_mask  # invert to find gaps

    return gaps

def find_largest_area(regions):
    """
    Add docstring
    """
    # Iterate over regions until the region with the largest area is found
    largest_area = 0
    largest_area_region = None
    if not regions:
        return largest_area, largest_area_region
    else:
        max_intensity_all_regions = max([region.intensity_max for region in regions])
        for region in regions:
            if int(region.intensity_max) == int(max_intensity_all_regions) and region.area > largest_area:
                largest_area = region.area
                largest_area_region = region
        return largest_area, largest_area_region

def find_largest_gap(mask):
    """
    Add docstring
    gaps (numpy array): binary image where background columns are set to 1.
    """
    # Find gaps
    horizontal_gaps = find_gaps_horizontally(mask)
    vertical_gaps = find_gaps_vertically(mask)

    regions_horizontal_gaps = measure_properties(horizontal_gaps)
    regions_vertical_gaps = measure_properties(vertical_gaps)

    # Determine in which orientation the largest gap is
    largest_area_horizontal, largest_area_region_horizontal = find_largest_area(regions_horizontal_gaps)
    largest_area_vertical, largest_area_region_vertical = find_largest_area(regions_vertical_gaps)

    if largest_area_vertical > largest_area_horizontal:
        gap_orientation = 'vertical'
        largest_gap = int(largest_area_region_vertical.centroid[0])
        return largest_gap, gap_orientation
    elif largest_area_horizontal > largest_area_vertical:
        gap_orientation = 'horizontal'
        largest_gap = int(largest_area_region_horizontal.centroid[1])
        return largest_gap, gap_orientation
    else:
        # When both 0 (probably) -> raise Error, warning or pass? This would mean no foreground was detected at all
        # raise ValueError("No gaps found.")
        warnings.warn("No gaps found, proceed with original mask", UserWarning)
        return mask

def find_dividing_line(mask):
    """
    Add docstring
    mask: binary mask
    """
    mask = crop_to_contour(mask)  # crop to contour tissue
    largest_gap, orientation_diving_line = find_largest_gap(mask)  # find largest gap in landscape and portrait orientation

    # TO DO: adjust dividing line so that dividing_line =  delta mask width-crop x

    # Check if largest region is found.
    if largest_gap != 0:
        dividing_line = largest_gap
    else:
        # If no gap found, use middle
        dividing_line = mask.shape[1] // 2
        orientation_diving_line = 'horizontal'

    return dividing_line, orientation_diving_line

def retrieve_objects_count_per_side(mask, regions):
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
    dividing_line, orientation_dividing_line = find_dividing_line(mask)
    dimension = 0 if orientation_dividing_line == 'vertical' else 1

    for region in regions:
        # Classify based on position
        if region.centroid[dimension] < dividing_line:
            left_objects_count += 1
        else:
            right_objects_count += 1

    return left_objects_count, right_objects_count

def retrieve_circularities_per_side(mask, regions):
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
    dividing_line, orientation_dividing_line = find_dividing_line(mask)
    dimension = 0 if orientation_dividing_line == 'vertical' else 1

    for region in regions:
        # Calculate circularity
        circularity = 4 * np.pi * region.area / region.perimeter ** 2

        # Classify based on position
        if region.centroid[dimension] < dividing_line:
            left_circularities.append(circularity)
        else:
            right_circularities.append(circularity)

    return left_circularities, right_circularities

def remove_control_pixels(mask, orientation, dividing_line, side_to_keep):
    """
    Add docstring
    side (character): 'left/top', 'right/bottom'
    """
    width = mask.shape[1]
    height = mask.shape[0]
    mask_without_controls = np.zeros_like(mask)
    if orientation == 'vertical':
        if side_to_keep == 'left/top':
            mask_without_controls[0:dividing_line, 0:width] = mask[0:dividing_line, 0:width]
        elif side_to_keep == 'right/bottom':
            mask_without_controls[dividing_line:height, 0:width] = mask[dividing_line:height, 0:width]
    if orientation == 'horizontal':
        if side_to_keep == 'left/top':
            mask_without_controls[0:height, 0:dividing_line] = mask[0:height, 0:dividing_line]
        elif side_to_keep == 'right/bottom':
            mask_without_controls[0:height, dividing_line:width] = mask[0:height, dividing_line:width]

    return mask_without_controls

def max_circularity_method(mask, regions):
    """
    Add docstring
    """
    # Determine whether controls are located left or right
    left_circularities, right_circularities = retrieve_circularities_per_side(mask, regions)

    # Decide which side to include based on mean circularity
    max_circularity_left = max(left_circularities)
    max_circularity_right = max(right_circularities)

    if max_circularity_left > max_circularity_right:
        # Include objects on the left side
        side_to_keep = 'right/bottom'
    elif max_circularity_left < max_circularity_right:
        # Include objects on the right side
        side_to_keep = 'left/top'
    return side_to_keep

def max_area_method(mask, regions):
    """
    Add doxstring
    """
    dividing_line, orientation_dividing_line = find_dividing_line(mask)
    dimension = 0 if orientation_dividing_line == 'vertical' else 1

    # Retrieve object area on each side
    for region in regions:
        if region.centroid[dimension] < dividing_line:
            left_area = region.area
        else:
            right_area = region.area

    # It is assumed the object with a higher area is the biopsy
    if right_area > 1.5 * left_area:
        side_to_keep = 'left/top'
    elif left_area > 1.5 * right_area:
        side_to_keep = 'right/bottom'
    else:
        side_to_keep = max_circularity_method(mask, regions)
    return side_to_keep

def counting_method(mask, objects_left, objects_right):
    """
    Add docstring -> assumes that there are 3-5 controls on the slide
    """
    # Decide which side to include using a range for the number of objects on each side
    if (objects_left >= 3 and objects_left <= 5) and (objects_right == 1):
        # Include objects on the left side
        side_to_keep = 'right/bottom'
        return side_to_keep
    elif (objects_right >= 3 and objects_right <= 5) and (objects_left == 1):
        # Include objects on the right side
        side_to_keep = 'left/top'
        return side_to_keep
    else:
        return False

def median_circularity_method(mask, regions):
    """
    Add docstring
    """
    # Determine whether controls are located left or right
    left_circularities, right_circularities = retrieve_circularities_per_side(mask, regions)

    # Decide which side to include based on mean circularity
    median_circularity_left = np.median(left_circularities)
    median_circularity_right = np.median(right_circularities)

    if median_circularity_left > median_circularity_right:
        # Include objects on the left side
        side_to_keep = 'right/bottom'
    elif median_circularity_left < median_circularity_right:
        # Include objects on the right side
        side_to_keep = 'left/top'
    return side_to_keep

def remove_controls_per_method(mask, temp_mask=None):
    """
    Add docstring
    """
    if temp_mask is None:
        temp_mask = mask

    # Measure properties of labeled regions
    regions = measure_properties(temp_mask)

    # If second small objects filter is too stringent, then do not filter
    objects_left, objects_right = retrieve_objects_count_per_side(temp_mask, regions)  # left == top, right == bottom
    if objects_left == 1 and objects_right == 1:
        side_to_keep = max_area_method(temp_mask, regions)
    else:
        # If objects on one single side is within the default number of controls (3-5) then that side contains controls
        side_to_keep = counting_method(temp_mask, objects_left, objects_right)
        if side_to_keep != False:
            pass
        else:
            side_to_keep = median_circularity_method(temp_mask, regions)

    # Remove controls
    dividing_line, orientation_dividing_line = find_dividing_line(temp_mask)
    mask = remove_control_pixels(mask, orientation_dividing_line, dividing_line, side_to_keep)

    return mask

def remove_controls(mask):
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
    >>> result_objects, circularities_all, circularities_included = extract_round_objects(input_mask)
    """
    # Second small objects filter (permitted here because only controls are needed)
    temp_mask = remove_small_objects(mask, metric='median', range_scale=0.08)

    # Measure properties of labeled regions
    regions = measure_properties(temp_mask)

    # If second small objects filter is too stringent, then do not filter
    objects_left, objects_right = retrieve_objects_count_per_side(temp_mask, regions)  # left == top, right == bottom
    if objects_left == 0 or objects_right == 0:
        warnings.warn("Zero objects detected on left/right side. Original mask will be used to remove controls", UserWarning)
        mask = remove_controls_per_method(mask=mask, temp_mask=None)
        return mask
    else:
        mask = remove_controls_per_method(mask=mask, temp_mask=temp_mask)
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


##################
### MRXS files ###
##################
INPUT_PATH = r'C:\Users\n.ultee\PycharmProjects\ER_ITH_v1.0\IRBm19_185_ER\mrxs_files3'

####
images = []
slide_images = []
masks = []
regionprops_controls = []
tiled_images = []
# Iterate over each file in the folder
for filename in os.listdir(INPUT_PATH):
    FILE_PATH = os.path.join(INPUT_PATH, filename, (filename + '.mrxs'))

    # Check if it's a regular file
    if os.path.isfile(FILE_PATH):
        print(FILE_PATH)

    # # Load image using dlup
    # slide_image = SlideImage.from_file_path(FILE_PATH)
    # slide_images.append(slide_image)
    #
    # # Convert image to numpy array
    # max_slide = max(slide_image.size)
    # size = max_slide * slide_image.mpp / 10  # Size is 10 mpp
    # size = int(max([int(1 if int(size) == 0 else 2 ** (int(size) - 1).bit_length()), 512]))
    # image = np.asarray(slide_image.get_thumbnail(size=(size, size)))
    # images.append(image)  # remove later

    # Load image using openslide
    slide_image = OpenSlide(FILE_PATH)
    slide_images.append(slide_image)

    # Crop to slide bounds (at level 5)
    image = crop_to_slide_bounds(slide_image, 0, 5)
    images.append(image)  # remove later

    # Generate mask
    mask = generate_mask(image)
    masks.append(mask)

    # Detect controls by round object detection (regionprops)
    round_objects = remove_controls(mask)
    regionprops_controls.append(round_objects)

    # # Tile image
    # TARGET_MPP = 100  # microns per pixel
    # TILE_SIZE = (10, 10)  # does not cause bg detection problem
    # tiled_image = tile_image(FILE_PATH, TARGET_MPP, TILE_SIZE, mask)
    # tiled_images.append(tiled_image)  # remove later

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

#################
### SVS files ###
#################
INPUT_PATH = r'C:\Users\n.ultee\PycharmProjects\ER_ITH_v1.0\IRBm19_185_ER\svs_files'

images = []
tiled_images = []
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
    slide_images.append(slide_image)

    # Convert image to numpy array
    max_slide = max(slide_image.size)
    size = max_slide * slide_image.mpp / 10  # Size is 10 mpp
    size = int(max([int(1 if int(size) == 0 else 2 ** (int(size) - 1).bit_length()), 512]))
    image = np.asarray(slide_image.get_thumbnail(size=(size, size)))
    images.append(image)  # remove later

    # Generate mask
    mask = generate_mask(image)
    masks.append(mask)

    # Detect controls by round object detection (regionprops)
    round_objects = remove_controls(mask)
    regionprops_controls.append(round_objects)

    # TO DO: when no controls can be determined, do not automatically take right/bottom

    # # Tile image
    # TARGET_MPP = 100  # microns per pixel
    # TILE_SIZE = (10, 10)  # does not cause bg detection problem
    # tiled_image = tile_image(FILE_PATH, TARGET_MPP, TILE_SIZE, mask)
    # tiled_images.append(tiled_image)  # remove later

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

########################################################################################################################
################################################## UNDER CONSTRUCTION ##################################################
########################################################################################################################

################
### Cellpose ###
################
import openslide
from cellpose import models, plot
from csbdeep.utils import normalize
INPUT_PATH = r'C:\Users\n.ultee\PycharmProjects\ER_ITH_v1.0\IRBm19_185_ER\svs_files'
# filename = r'T18-02695 I1 ER.svs'
# filename = r'T17-23462 B2 ER.svs'
filename = r'T17-21493 B1 ER.svs'
slide = openslide.OpenSlide(os.path.join(INPUT_PATH, filename))
# slide = slide_images[7]

# Get information about the levels
levels = slide.level_count
print(f"Number of levels: {levels}")

# Choose the level for analysis (e.g., level 1)
chosen_level = 0

# Read the image at the chosen level
# image_at_level = np.array(slide.read_region((0, 0), chosen_level, slide.level_dimensions[chosen_level]))[:, :, :3]

# Specify the region of interest (ROI) coordinates
# roi_x, roi_y = 6000, 20000  # Example coordinates
# roi_width, roi_height = 1000, 1000  # Example dimensions
# roi_x, roi_y = 2000, 26666  # Example coordinates T17-23462 B2 ER
# roi_width, roi_height = 1000, 1000  # Example dimensions T17-23462 B2 ER
roi_x, roi_y = 4300, 13850  # Example coordinates T17-21493 B1 ER
roi_width, roi_height = 300, 300  # Example dimensions T17-21493 B1 ER

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

# Cellpose
model = models.Cellpose(gpu=True, model_type='cyto')
masks, flows, styles, diams = model.eval(gray_image, diameter=None, channels=[0, 0])

fig = plt.figure(figsize=(12, 5))
plot.show_segmentation(fig, gray_image, masks, flows[0], channels=[0, 0])
plt.tight_layout()
plt.show()

### Use masks to find cells in original image
from cv2 import split, threshold, THRESH_BINARY, THRESH_OTSU

def assess_er_positivity(image, mask):
    """
    Add docstring
    """
    average_red_values = []
    labeled_mask, num_labels = measure.label(mask, connectivity=2, return_num=True)  # label objects
    for label in np.unique(labeled_mask):
        if label == 0:
            continue

        object_mask = np.where(labeled_mask == label, 1, 0)
        object_pixels = image[object_mask == 1]

        # Calculate the average red channel value for the current object
        average_red = np.mean(object_pixels[:, 2])  # Assuming the red channel is at index 2

        # Store the result in the dictionary
        average_red_values.append(average_red)

    image_masked = image[:, :, 0].copy()
    image_masked[~(mask > 0)] = 0

    total_cells = len(average_red_values)
    er_positive_cells = sum(1 for value in average_red_values if value < 165)
    er_percentage = int(er_positive_cells / total_cells * 100)
    print(f'ER: {er_percentage}% ({er_positive_cells}/{total_cells})')

    return er_percentage

assess_er_positivity(image_roi, masks)

### Try on whole IHC image
# Cellpose only works on higher level (higher resolution)
image = np.copy(images[7])
mask = np.copy(regionprops_controls[7])
for channel in range(3):  # Assuming RGB image (3 channels)
    image[:, :, channel] = np.where(mask, image[:, :, channel], 0)
    # Convert the RGB image to grayscale if needed
    sd_image = normalize(image.astype(np.float32))

    gray_image = 1 - np.mean(sd_image, axis=-1)
    plt.imshow(gray_image, cmap="gray")
    plt.show()

    # Cellpose
    model = models.Cellpose(gpu=True, model_type='cyto')
    masks, flows, styles, diams = model.eval(gray_image, diameter=None, channels=[0, 0])

    er_percentage = assess_er_positivity(image, masks)

### Try on multiple IHC images
# Cellpose only works on higher level (higher resolution)
er_percentages = []
for image_id in (range(0, len(images))):
    # Remove controls
    image = np.copy(images[image_id])
    mask = np.copy(regionprops_controls[image_id])
    for channel in range(3):  # Assuming RGB image (3 channels)
        image[:, :, channel] = np.where(mask, image[:, :, channel], 0)

    # Convert the RGB image to grayscale if needed
    sd_image = normalize(image.astype(np.float32))

    gray_image = 1 - np.mean(sd_image, axis=-1)
    plt.imshow(gray_image, cmap="gray")
    plt.show()

    # Cellpose
    model = models.Cellpose(gpu=True, model_type='cyto')
    masks, flows, styles, diams = model.eval(gray_image, diameter=None, channels=[0, 0])

    er_percentage = assess_er_positivity(image, masks)
    er_percentages.append(er_percentage)



### Binarize image using Red channel only
from cv2 import split, threshold, THRESH_BINARY, THRESH_OTSU

# # Combine masks for individual cells into one mask
# combined_mask = np.sum(masks, axis=0)
#
# # Convert the combined mask to binary (0 for background, 1 for cells)
# binary_mask = combined_mask > 0

red_channel = image_roi[:, :, 0]
isolated_cells = red_channel * masks

plt.imshow(isolated_cells)
plt.show()

# Threshold the red channel
_, thresholded_red = threshold(red_channel, 0, 255, THRESH_BINARY + THRESH_OTSU)

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(image_roi)
plt.title('Original Image')
plt.subplot(132)
plt.imshow(isolated_cells, cmap='gray')
plt.title('Isolated Cells')
plt.subplot(133)
plt.imshow(thresholded_red, cmap='gray')
plt.title('Thresholded Red Channel')
plt.tight_layout()
plt.show()

cells_total = np.sum(masks)
er_positive = np.sum(thresholded_red > 0)

################ GRAVEYARD ################
differences_per_mask = []

for image in images:
    # Convert RGB image to grayscale
    image = image.copy()
    gray_image = color.rgb2gray(image)

    # Perform dilation and erosion operations
    dilated_img = dilation(gray_image, square(10))
    eroded_img = erosion(gray_image, square(3))
    dilated_min_eroded_img = dilated_img - eroded_img

    # find good filter for segmentation
    # fig, ax = try_all_threshold(dilated_min_eroded_img, figsize=(10, 8), verbose=False)
    # plt.show()

    # Thresholding to create a binary mask
    threshold = threshold_mean(dilated_min_eroded_img)  # set threshold to segment (yes/no)
    mask = dilated_min_eroded_img >= threshold  # dark pixels==False, bright pixels==True

    # # Remove small objects from the binary mask
    # mask = remove_small_objects(mask, metric='maximum', percentage_max_area=0.02)
    plt.imshow(mask, cmap='gray')
    plt.show()

    labeled_mask, num_labels = measure.label(mask, connectivity=2, return_num=True)

    # Measure properties of labeled regions
    regions = measure.regionprops(labeled_mask)

    # Sort the areas
    areas = [region.area for region in regions]
    sorted_areas = sorted(areas)
    # Calculate the differences between consecutive elements
    differences = np.diff(sorted_areas)
    differences_per_mask.append(differences)

    plt.boxplot(differences)
    plt.show()

# Create a boxplot
plt.boxplot(differences_per_mask)
plt.xlabel('Mask Index')
plt.ylabel('Difference in Areas')
plt.title('Boxplot of Differences between Consecutive Areas for Each Mask')
plt.show()

### read_region
FILE_PATH = r'C:\\Users\\n.ultee\\PycharmProjects\\ER_ITH_v1.0\\IRBm19_185_ER\\mrxs_files\\T20-05125 I4 ER\\T20-05125 I4 ER.mrxs'
slide_image = SlideImage.from_file_path(FILE_PATH)

location, size = slide_image.slide_bounds

downsample_factor = 32  # level 0 to 5
location = tuple(value // downsample_factor for value in location)

image_at_level = np.array(slide_image.read_region((5195, 976), 5, (97130, 170030)))  # memory error

def determine_orientation(mask):
    """
    Add docstring
    """
    width = mask.shape[1]
    height = mask.shape[0]
    if width > height:
        original_orientation = 'landscape'
        landscape = mask
        portrait = np.rot90(mask)
    else:
        original_orientation = 'portrait'
        landscape = np.rot90(mask)
        portrait = mask

    return original_orientation, landscape, portrait

def find_largest_gap_in_both_dimensions(mask):
    """
    Add docstring
    """
    # Determine original orientation
    original_orientation, landscape, portrait = determine_orientation(mask)

    # Find the largest gap in both orientations
    largest_area_horizontal, region_largest_gap_horizontal = find_largest_gap(landscape)
    largest_area_vertical, region_largest_gap_vertical = find_largest_gap(portrait)

    # Return centroid largest gap
    if largest_area_horizontal > largest_area_vertical:
        return int(region_largest_gap_horizontal.centroid[1])
    else:
        return int(region_largest_gap_vertical.centroid[0])

    # # Return centroid largest gap according to original orientation
    # if original_orientation == 'landscape':
    #     if largest_area_horizontal > largest_area_vertical:
    #         return int(region_largest_gap_horizontal.centroid[1])
    #     else:
    #         return int(region_largest_gap_vertical.centroid[0])
    # elif original_orientation == 'portrait':
    #     if largest_area_horizontal > largest_area_vertical:
    #         return int(region_largest_gap_horizontal.centroid[0])
    #     else:
    #         return int(region_largest_gap_vertical.centroid[1])

##################
### MRXS FILES ###
##################
image = images[0]
plt.imshow(image)
plt.show()

print(image.shape)  # (7577, 8192, 3)
print(image.dtype)  # uint8

plt.imshow(image[7000:7010, 0:10])
plt.show()
print(image[7000:7010, 0:10])

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
# INPUT_PATH = r'C:\Users\n.ultee\PycharmProjects\ER_ITH_v1.0\IRBm19_185_ER\svs_files'
# filename = r'T18-02695 I1 ER.svs'
INPUT_PATH = r'C:\Users\n.ultee\PycharmProjects\ER_ITH_v1.0\IRBm19_185_ER\mrxs_files\T20-05125 I4 ER'
filename = r'T20-05125 I4 ER.mrxs'
slide = openslide.OpenSlide(os.path.join(INPUT_PATH, filename))

# Get information about the levels
levels = slide.level_count
print(f"Number of levels: {levels}")

# Choose the level for analysis (e.g., level 1)
chosen_level = 5

# Read the image at the chosen level
image_at_level = np.array(slide.read_region((0, 0), chosen_level, slide.level_dimensions[chosen_level]))[:, :, :3]
plt.imshow(image_at_level)
plt.show()

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

# Load the pre-trained Stardist model
model = StarDist2D.from_pretrained('2D_paper_dsb2018')

# Predict probabilities and labels
# labels, probabilities = model.predict(gray_image)  # label -> radial distance; probability -> pixel/point is center
labels, _ = model.predict_instances(normalize(gray_image))

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap="gray")
plt.axis("off")
plt.title("input image")

plt.subplot(1, 2, 2)
plt.imshow(render_label(labels, img=gray_image))
plt.axis("off")
plt.title("prediction + input overlay")

plt.show()

### Contrast stretching
from skimage.exposure import equalize_adapthist, rescale_intensity

# Try transformations
contrast_stretched = rescale_intensity(gray_image, in_range='image', out_range=(0, 1))  # contrast stretching

normalized_image = gray_image / gray_image.max()
contrast_enhanced = equalize_adapthist(normalized_image)  # adaptive histogram equalization

# plot
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(contrast_enhanced, cmap='gray')
plt.title('Contrast Stretched Image')

plt.show()

# Cellpose with transformed image
model = models.Cellpose(gpu=True, model_type='cyto')
masks, flows, styles, diams = model.eval(contrast_enhanced, diameter=None, channels=[0, 0])

fig = plt.figure(figsize=(12, 5))
plot.show_segmentation(fig, contrast_enhanced, masks, flows[0], channels=[0, 0])
plt.tight_layout()
plt.show()

### Access outlines
from cellpose import utils
outlines = utils.masks_to_outlines(masks)
plt.imshow(outlines, cmap='gray')
plt.show()

# Label and measure properties of the objects
regions = measure_properties(outlines)
len(regions)  # Take a look at the 50x50 image and you will see that the outlines are connected so 3 objects yielded...

### Labels method
# Apply threshold
binary_mask = (masks > 0).astype(int)

# Label cells
labeled_cells, num_cells = ndimage.label(binary_mask)

# Visualize if needed
plt.imshow(image_roi)
plt.contour(labeled_cells, colors='red', linewidths=1, alpha=0.5)
plt.show()

# Print or use the count
print(f"Number of cells: {num_cells}")

