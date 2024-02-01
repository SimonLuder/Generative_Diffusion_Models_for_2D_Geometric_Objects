import os
from pathlib import Path
import shutil
import re

def sort_images(base_dir, target_dir, regex_patterns):
    """
    Sorts images into subfolders based on a substring in the filename.

    Args:
        base_dir (str): The directory containing the images.
        num_folders (int): The number of subfolders to create.
    """

    # Create the subfolders
    for pattern in regex_patterns:
        Path(os.path.join(target_dir, str(pattern))).mkdir(parents=True, exist_ok=True)

    # Move the images to the appropriate subfolder
    for filename in os.listdir(base_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # add more extensions if needed

            for pattern in regex_patterns:
                if re.search(pattern, filename):
                    # Move the file
                    shutil.copy(os.path.join(base_dir, filename), os.path.join(target_dir, str(pattern), filename))
                    break  # a file is moved to the first matching pattern's folder

if __name__ == "__main__":
    base_dir = "../data/Unconditional/shapes32"
    target_dir = "../data/Conditional/"
    regex_patterns = ["Triangle", "Circle", "Heptagon", "Octagon", "Hexagon", "Square", "Nonagon", "Pentagon", "Star"]
    sort_images(base_dir, target_dir, regex_patterns)