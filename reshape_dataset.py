import os
import cv2
import argparse
import pandas as pd
from tqdm import tqdm


def resize_images(source_folder, dest_folder, size=(128, 128)):
    """
    Resizes images from a source folder and saves them to a destination folder.

    This function reads each image file (with .jpg or .png extension) from the source folder,
    resizes it to the specified size using area interpolation, and writes the resized image
    to the destination folder. If the destination folder does not exist, it is created.

    Args:
        source_folder (str): Path to the source folder containing the images.
        dest_folder (str): Path to the destination folder where resized images will be saved.
        size (tuple, optional): Desired size of the output images as a tuple (width, height). 
            Default is (128, 128).

    Returns:
        None
    """
    if not os.path.exists(os.path.join(dest_folder, "images")):
        os.makedirs(os.path.join(dest_folder, "images"))

    for filename in tqdm(os.listdir(os.path.join(source_folder, "images"))):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(source_folder, "images", filename))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(dest_folder, "images", filename), img)


def replace_image_paths(df, column, new_dir):
    """
    Replace the directory part of the file paths in a DataFrame but keep the filenames.

    Args:
        df (DataFrame): DataFrame containing the file paths.
        column (str): Column name containing the file paths.
        new_dir (str): New directory to replace the old one.

    Returns:
        DataFrame: DataFrame with updated file paths.
    """
    df[column] = df[column].apply(lambda x: os.path.join(new_dir, os.path.basename(x)))
    df[column] = df[column].str.replace("\\", "/")
    return df


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="./data/train/", help="Source directory")
    parser.add_argument('--destination', type=str, default=f"./data/train32")
    parser.add_argument('--custom_destination_path', type=str, default=None)
    parser.add_argument('--size', type=int, default=32, help="Image size for square proportions")
    args = parser.parse_args()

    if not os.path.exists(args.destination):
        os.makedirs(args.destination)

    # update and save csv
    if args.custom_destination_path is not None:
        destination_path = args.custom_destination_path
    else:
        destination_path = args.destination

    df = pd.read_csv(os.path.join(args.source, "labels.csv"))
    df = replace_image_paths(df, "destination", destination_path)
    df = replace_image_paths(df, "file", os.path.join(destination_path, "images"))
    df.to_csv(os.path.join(args.destination, "labels.csv"), index=False)

    # reshape and move images
    resize_images(source_folder=args.source, dest_folder=args.destination, size=(args.size, args.size))