import os
import cv2
from collections import defaultdict

def extract_prefix(file_name):
    """
    Extract the prefix from the filename, removing '_MS_' or '_Aligned' if present.
    Args:
        file_name (str): Name of the file.
    Returns:
        str: Extracted prefix without '_MS_' or '_Aligned'.
    """
    if '_MS_' in file_name or '_Aligned' in file_name:
        return '_'.join(file_name.split('_')[:3])  # e.g., "DJI_20240807124101_0006"
    return None

def load_image_set(files):
    """
    Load a set of images into a list ordered by band type.
    Args:
        files (list): List of file paths for the image set.
    Returns:
        list: A list of 5 images in the order [green, red, NIR, red-edge, RGB].
    """
    image_set = [None] * 5
    for file in files:
        if '_MS_G' in file:
            image_set[0] = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        elif '_MS_R' in file:
            image_set[1] = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        elif '_MS_NIR' in file:
            image_set[2] = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        elif '_MS_RE' in file:
            image_set[3] = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        elif '_Aligned' in file:
            image_set[4] = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    return image_set

def process_all_sets(directory):
    """
    Process all sets of images in the given directory.
    Args:
        directory (str): Path to the directory containing the images.
    """
    # List all files in the directory
    file_paths = sorted([os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    
    # Group filenames by their shared prefix
    grouped_files = defaultdict(list)
    for file in file_paths:
        base_name = os.path.basename(file)
        prefix = extract_prefix(base_name)
        if prefix:
            grouped_files[prefix].append(file)
    
    # Iterate over each group and process one set at a time
    for prefix, files in grouped_files.items():
        if len(files) == 5:  # Ensure the group has exactly 5 files
            image_set = load_image_set(files)
            # Perform processing on this set
            print(f"Processing set with prefix: {prefix}")
            # Example: Print the shape of each image
            for idx, img in enumerate(image_set):
                if img is not None:
                    print(f"Image {idx + 1}: {img.shape}")
        else:
            print(f"Skipping incomplete set with prefix: {prefix}")

if __name__ == "__main__":
    # Directory containing your images
    image_directory = "/path/to/your/images"
    process_all_sets(image_directory)



# Open the file in append mode if it exists, otherwise in write mode
    with open(file_name, 'a' if file_exists else 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # If the file doesn't exist, write the header first
        if not file_exists:
            writer.writerow(header)
        
        # Add the new row of data
        writer.writerow(row)

