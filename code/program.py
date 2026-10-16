import cv2
import numpy as np
import os
from collections import defaultdict
import csv
import matplotlib.pyplot as plt

'''
extract_prefix: Extracts the prefix from the image name to be used to group related images

Input: Image name
Output: Prefix of image that will be used to group related images
	or none if name pattern is not identifyed

'''
def extract_prefix(file_name):
	# Use naming pattern to identify where the prefix ends
	if '_MS_' in file_name or '_Aligned' in file_name:

		# Splits name and rejoins the need portion of the image name
		return '_'.join(file_name.split('_')[:3])  # for example "DJI_20240807124101_0006"

	# If name pattern is not identifyed return none
	return None

'''
load_image_set: Loads in each group of images for the current loaction being examined
	each image will be added to a list of images in the order green band, red band,
	NIR band, Red-edge band, and RGB image

Input: A list of the directories paths to each image (this includes the image name)
Output: A list of the 5 related images in the order: [green, red, NIR, red-edge, RGB]

'''
def load_image_set(files):
	# Create empty list to store images
    image_set = [None] * 5

    # Loop over the files and add each to the list in order
    for file in files:
    	if '_MS_G' in file:
    		image_set[0] = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    	elif '_MS_RE' in file:
    		image_set[3] = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    	elif '_MS_NIR' in file:
    		image_set[2] = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    	elif '_MS_R' in file:
    		image_set[1] = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    	elif '_Aligned' in file:
    		image_set[4] = cv2.imread(file, cv2.IMREAD_UNCHANGED)

    # Return the list of images
    return image_set
'''
apply_mask: Applys a binnary mask to a given image, 
	by using the bitwise 'and' operator to create a new image
	that has all the pixels set to zero that were set to zero in the mask

Input: The image to be filterd by the mask
Input: The mask to be used to filter the image
output: the filtered image

'''
def apply_mask(image, mask):
	# Check if Image and Mask are the same type
	if mask.dtype != image.dtype:

		# Change mask to be the same type as image
		mask = mask.astype(image.dtype)

	# Confirm that image and mask are both not None and are the same shape
	if mask is not None and image is not None and mask.shape == image.shape:

		# Create a new image that is the filtered image
		masked_image = cv2.bitwise_and(image, mask)

	return masked_image

'''
water_mask: Uses the Red-edge band to create a binary mask to remove water from the 
	images by applying a threshold where all pixels below the threshold are set to
	0 and all values above the threshold are set to the max value in the image 
	(for a 16 bit image the max value equals 65535)

Input: The red-edge band image after being resize
Input: Optionally, the threshold value can be inputed or use the default of 11500
Output: The water mask or 
	throws error if image is None or not in 16-bit format 

'''
def water_mask(image, threshold_value=11500):
	# Checks if the image is not None and is a 16 bit image
	if image is not None and image.dtype == np.uint16:

		# Creates binary water mask and then returns it
		_, mask = cv2.threshold(image, threshold_value, 65535, cv2.THRESH_BINARY)
		return mask

	# If image is None or not a 16 bit image an error is thrown
	else:
		raise ValueError("Error: Image not loaded properly or not in 16-bit format.")
'''
resize_images: Upscales the spectral band images to be the same size as the RGB image

input: The set of images for the location being examined
output: The set of images with the upscaled specral bands or 
	throws error if RGB iamge is missing

'''
def resize_images(image_set):
	# Check that the RGB image is in the list
	if image_set[4] is None:

		# If the image is missing throw error
		raise ValueError("The RGB image at index 4 (reference image) is not loaded.")

	# Resize image
	else:
		# Get the size of the RGB image
		height, width = image_set[4].shape[:2]

		# Loop over spectral band images to resize
		for i in range(4):
			if image_set[i] is not None:
				image_set[i] = cv2.resize(image_set[i], (width, height), interpolation=cv2.INTER_LINEAR)

		# Return image set with spectral band images resized
		return image_set
'''
split_RGB: Splits the RGB image in to three gray scale images and puts them in a list 
	in the order Red, Green, Blue.

input: The RGB image to be split in to the three channels
output: A list containing the three channels from the RGB image,
	with red in index 0, green in index 1, and blue in index 2

'''
def split_RGB(image):
	# Create a list of size 3 to put images in
	image_set = [None] * 3

	# Split RGB iamge
	blue_channel, green_channel, red_channel = cv2.split(image)

	# Add the channels to the image list and returns it
	image_set[0] = red_channel
	image_set[1] = green_channel
	image_set[2] = blue_channel

	return image_set
'''
nroml_image: Normalizes the kelp portion of the image to have pixel values between 0.0 and 1.0, 
	doing this minimizes classification errors caused by different lighting in images. This process also
	enhances the contrast by stretching the intensity range making sudtle details more visible.

Input: The image to be normlized (water already filted out) 
Input: Optional, the minimum value in the new pixel range. (default set to 0)
Input: Optional, the maximum value in the new pixel range. (default set to 1)
Input: Optional, the type of normalization used. (default set to cv2.NORM_MINMAX)
Output: The normalized image 

'''
def nroml_image(image, min_value=0, max_value=1, norm_type=cv2.NORM_MINMAX):
	# Create and apply a boolean mask to remove background pixels (water)
	non_zero_mask = image > 0
	non_zero_values = image[non_zero_mask]

	# Checks that that non_zero_values is not empty
	if len(non_zero_values) > 0:

		# Normalize non-zero pixels in image
		normalized_values = cv2.normalize(non_zero_values, None, min_value, max_value, norm_type, cv2.CV_32F).flatten()

		# Create a copy of the image and convert to floating point
		normalized_image = image.astype(np.float32).copy()

		# Assige Normalize non-zero-pixels back to image
		normalized_image[non_zero_mask] = normalized_values
	else:

		# If no non-zero-pixels, convert image to floating point
		normalized_image = image.astype(np.float32)

	return normalized_image

'''
RGB_masks_8bit: Creates masks for the red, Green and blue channels of the RGB image to identify bryozoan on kelp
	tissue. Applies a threshold range to each image to create the maskes. 

Inputs: The three channels from the RGB image (must be normalized and have the water filtered out)
Output: A list of the bryozoan masks for each image in the order [red, green, blue] or
	throws error if one of the three images are not loeaded properly or not in floating point format

'''
def RGB_masks_8bit(red_normalized_image, green_normalized_image, blue_normalized_image):
	# Create list for masks
	RGB_mask = [None]*3

	# Check that the image exist and is in floating point format 
	if red_normalized_image is not None and red_normalized_image.dtype == np.float32:

		# Create upper and lower bounds for threshold range
		lower = 0.77
		upper = 0.9

		# Create mask using the threshold range set above
		mask = cv2.inRange(red_normalized_image, lower, upper)

		# Add mask to proper index in list
		RGB_mask[0] = mask

	# Check image 
	if green_normalized_image is not None and green_normalized_image.dtype == np.float32:

		# Create range and mask. Then add mask to list
		lower = 0.7 
		upper = 0.86
		mask = cv2.inRange(green_normalized_image, lower, upper) 
		RGB_mask[1] = mask

	# Check image 
	if blue_normalized_image is not None and blue_normalized_image.dtype == np.float32:

		# Create range and mask. Then add mask to list
		lower = 0.72
		upper = 0.85
		mask = cv2.inRange(blue_normalized_image, lower, upper)  
		RGB_mask[2] = mask

	else:
		# Throw error if problem with image arises
		raise ValueError("Error: Image not loaded properly or not in float32 format.")

	return RGB_mask

'''
remove_small_regions: Cleans bryozoan mask by removeing noise in it. It dose this by 
	identifying connected components in the mask and removes components that are smaller then the 
	minimum meaningful size.

Input: The binary mask to be cleaned 
Input: Optional, the minimum meaningful size. (default set to 200)
Output: the Cleaned binary mask, with objects of interest having a pixel value of 255
	and the background having pixels values of 0

'''
def remove_small_regions(mask, min_area=200):
	# Identify connected components and save the need stats 
	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

	# Create new image with the same size as the mask and set all pixels to zero
	cleaned_mask = np.zeros_like(mask, dtype=np.uint8)

	# Loop over each component (excluding the background)
	for i in range(1, num_labels):

		# Check if the size of the component is greater then or equal to min area
		if stats[i, cv2.CC_STAT_AREA] >= min_area:

			# Set component in new mask to white
			cleaned_mask[labels == i] = 255

	# Return Cleaned mask
	return cleaned_mask
'''
clean_mask: Refine and clean bryozoan masks by preforming morpholgical opening and removing noise in mask. 
	Creates a moving window (kernel) to examine small parts of the image at a time. In each window the morpholgical 
	opening transformation first shrinks white parts of the mask by eroding boundaries, 
	then expands the white regions to restore the shape of larger objects while still minimizing noise. 
	Then calls remove_small_regions to clean up noise that was missed during the morpholgical transformation

Input: The mask to be cleaned
Input: Optional, the size of the kernel (default set to 3)
Input: Optinoal, the minimum meaningful size of object (default set to 200)
Output: The cleaned mask

'''
def clean_mask(mask, kernel_size=3, min_area=200):
	# Create moving window for morpholgical transformation
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

	# Preform morpholgical transformation
	opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

	# Clean up excess noise in mask
	refined_mask = remove_small_regions(opened_mask, min_area=min_area)

	# Retunr cleaned mask
	return refined_mask

'''
kelp_count: Calculates the percent of kelp tissue present in the image by counting the amount of 
	pixels that have a value of zero (water) versus the amount of pixel that are not zero (kelp).
	The number of pixels are then used to calucate the percent of kelp in the image.

Input: The three channels of the RGB image in the order red, green, blue
Output: A list containing the number of pixels classified as water and as kelp, and the precent of kelp.
	These are in the order of water pixels, kelp pixels, and precent of kelp.

'''
def kelp_count(red, green, blue):
	# Get the amount of pixels in each image
	total_red_pixels = red.size
	total_green_pixels = green.size
	total_blue_pixels = blue.size

	# Create empty list for results
	kelp_amount = [None] * 3

	# Check if the three chennal images are the same size
	if(total_red_pixels == total_green_pixels and total_red_pixels == total_blue_pixels):

		# Get the number of pixels that equal zero (water)
		water_pixels = np.count_nonzero(red == 0)

		# Calculate the number of Kelp pixels
		kelp_pixels = total_red_pixels - water_pixels

		# Calucate the percent of kelp in the image
		kelp_percent = (kelp_pixels / total_red_pixels) * 100

		# Add the results to the list and return it
		kelp_amount[0] = water_pixels
		kelp_amount[1] = kelp_pixels
		kelp_amount[2] = kelp_percent
		return kelp_amount

	# If for some reason the three chennal images are different sizes
	# Calculate the Sum of the water pixels, kelp pixels and total pixels
	# Then find the mean of each to use to calucate the percentage of kelp
	else:
		image_set = [red, green, blue]
		sum_pixel = 0
		sum_total = 0
		sum_water = 0
		for i in image_set:
			total_pixel = i.size
			sum_total = sum_total + total_pixel
			water_pixel = np.count_nonzero(i == 0)
			sum_water = sum_water + water_pixel
			kelp_pixel = total_pixel - water_pixel
			sum_pixel = sum_pixel + kelp_pixel

		kelp_pixels = sum_pixel/3
		water_pixels = sum_water/3
		total = sum_total/3
		kelp_percent = (kelp_pixels / total) * 100

		kelp_amount[0] = water_pixels
		kelp_amount[1] = kelp_pixels
		kelp_amount[2] = kelp_percent
		return kelp_amount

'''
bryozoan_count: Calculates the percent of bryozoan present in the mask. 

Input: binary masks
Input: a List conatining the amount of kelp in the image
output: A list containing the number of pixels in the background (zero values), 
	the numer of pixels classified as bryozoan, and the percent of bryozoan, respectively.

'''
def bryozoan_count_single(image, kelp):
	total_pixels = image.size
	zero_pixels = np.count_nonzero(image == 0)
	bryozoan_pixels = total_pixels - zero_pixels
	kelp_pixels = kelp[1]

	bryozoan_percent = (bryozoan_pixels / total_pixels) * 100
	bryozoan_percent_on_kelp = (bryozoan_pixels / kelp_pixels) * 100

	bryozoan_amount = [zero_pixels, bryozoan_pixels, bryozoan_percent, bryozoan_percent_on_kelp]
	return bryozoan_amount

'''
bryozoan_count: Calculates the percent of bryozoan present in the mask. 
	Takes a mask that slightly over estimates the amount of bryozoan present and a 
	mask the slightly under estimates the amount of bryozoan and calculate the mean of
	the total number of pixels, pixels with values of zero, and pixels classified as bryozoan
	to calucated the percent of bryozoan found on the kelp tissue. 

Input: Two binary masks, one that is an over estimate of bryozoan and the other is an under estimate
	of bryozoan present in the image.
output: A list containing the number of pixels in the background (zero values), 
	the numer of pixels classified as bryozoan, and the percent of bryozoan, respectively.

'''
def bryozoan_count(over, under):
	# Create a list with the two masks in it
	mask = [over, under]

	# Create values to use in loop
	sum_total = 0
	sum_pixel = 0
	sum_zero = 0

	# Loop over masks
	for i in mask:

		# Get total number of pixels and sum up
		total_pixels = i.size
		sum_total = sum_total + total_pixels

		# Get number of pixels with a value of zero and sum up
		zero_pixels = np.count_nonzero(i == 0)
		sum_zero = sum_zero + zero_pixels

		# Get number of pixels classified as bryozoan and sum up
		bryozoan_pixels = total_pixels - zero_pixels
		sum_pixel = sum_pixel + bryozoan_pixels

	# Calucate means
	total = sum_total/2
	zero_pixels = sum_zero/2
	bryozoan_pixels = sum_pixel/2

	# Calucate percent of bryozoan present and add results to list
	bryozoan_percent = (bryozoan_pixels / total) * 100
	bryozoan_amount = [zero_pixels, bryozoan_pixels, bryozoan_percent]
	
	return bryozoan_amount

'''
csv_label_file: Creates a csv file and add results to it. the csv file contains the perfix of the image
	the number of pixels classified as kelp, water and bryozoan as while as the precent of the image that has
	kelp and bryozoan in it.

Input: The prefix of the current image 
Input: A list containing the percent of kelp and number of pixels that are classified as water and kelp
Input: A list containing the percent of bryozoan and the number of pixels classifed as background and bryozoan
output: None

'''
def csv_label_file(image_prefix, kelp_amount, bryozoan_amount):
	# Create name of CSV file
	file_name = "kelp_bryozoan_labels.csv"

	# Check if CSV file exists
	file_exists = os.path.exists(file_name)

	# Create list with header names and list with results 
	headers = ["Name", "numKelp", "numWater", "NumBryozoan", "precentKelp", "precentBryozoan"]
	row = [image_prefix, kelp_amount[1], kelp_amount[0], bryozoan_amount[1], kelp_amount[2], bryozoan_amount[2]]

	# Open the CSV file, in either append mode or write mode depending on if the file exists
	with open(file_name, 'a' if file_exists else 'w', newline='') as csv_file:

		# Create CSV writer object
		writer = csv.writer(csv_file)

		# If file dose not exist add headers
		if not file_exists:
			writer.writerow(headers)

		# Write row of results to csv file
		writer.writerow(row)

'''
process_all_sets: Loops over all images in directory, sorts them by name to group related images. 
	Then process one set of images at a time to calucate the percent of bryozoan in it and write the 
	results to a csv file.

input: The directory with all images needing processing in it
output: None

'''
def process_all_sets(directory):
	# expands path to ensure the directory can be found
	directory = os.path.expanduser(directory)

	# Collects all paths to files in the directory and sorts them alphabetically
	file_paths = sorted([os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

	# Creates an empty dictionary containing lists
	grouped_files = defaultdict(list)

	# Loop over the sorted list of file paths to file dictionary
	# the keys are the prefix of each image group and the value contails a list of 
	# the paths to the five images in each group
	for file in file_paths:

		# get full path to file
		base_name = os.path.basename(file)

		# Get image prefix
		prefix = extract_prefix(base_name)

		# Prefix was found group files by prefix
		if prefix:
			grouped_files[prefix].append(file)

	# Loop over each group of images
	for prefix, files in grouped_files.items():

		# Check group has five iamges
		if len(files) == 5:

			# Load, resize, and mask water out of images
			image_set = load_image_set(files)
			image_set = resize_images(image_set)
			maskWater = water_mask(image_set[3], 11500)

			# Split RGB image in to its three chennals and apply water mask
			rgb_channels = split_RGB(image_set[4])
			red = apply_mask(rgb_channels[0], maskWater)
			green = apply_mask(rgb_channels[1], maskWater)
			blue = apply_mask(rgb_channels[2], maskWater)

			# Calculate the percent of kelp in image
			kelp_amount = kelp_count(red, green, blue)

			# Normlizes the three RGB chennals
			red_normalized_image = nroml_image(red, 0, 1)
			green_normalized_image = nroml_image(green, 0, 1)
			blue_normalized_image = nroml_image(blue, 0, 1)

			# Create bryozoan masks for each chennal of the RBG image
			rgb_mask = RGB_masks_8bit(red_normalized_image, green_normalized_image, blue_normalized_image)

			# Clean masks
			red_mask = clean_mask(rgb_mask[0], 3, 200)
			green_mask = clean_mask(rgb_mask[1], 3, 200)
			blue_mask = clean_mask(rgb_mask[2], 3, 200)

			# Combined masks to make the bryozoan under estimating mask
			bryozoan_mask_under = cv2.bitwise_and(blue_mask, green_mask)
			bryozoan_mask_under = cv2.bitwise_and(bryozoan_mask_under, red_mask)

			# Calucate the percent of bryozoan in image
			total_bryozoan = bryozoan_count_single(bryozoan_mask_under, kelp_amount)
			
			# Add results to CSV file
			csv_label_file(prefix, kelp_amount, total_bryozoan)
			
		# If all five images where not found skip processing and prind image prefix
		else:
			print(f"Skipping incomplete set with prefix: {prefix}")

# main
if __name__ == "__main__":

	# Set directory containing images
	image_directory = "./EasyToSpot"

	# Check that directory exists, if not found exit program
	if not os.path.exists(image_directory):
		print(f"Error: The directory {image_directory} does not exist.")
		exit(1)

	# process all images in directory
	process_all_sets(image_directory)


