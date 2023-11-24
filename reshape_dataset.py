import os
import cv2
import argparse
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def resize_images(source_folder, dest_folder, size=(128, 128)):
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

    Parameters:
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
    parser.add_argument('--size', type=int, default=32, help="Image size for square proportions")

  
    args = parser.parse_args()

    if not os.path.exists(args.destination):
        os.makedirs(args.destination)

    # update and save csv
    df = pd.read_csv(os.path.join(args.source, "labels.csv"))
    df = replace_image_paths(df, "destination", args.destination)
    df = replace_image_paths(df, "file", os.path.join(args.destination, "images"))
    df.to_csv(os.path.join(args.destination, "labels.csv"))

    resize_images(source_folder=args.source, dest_folder=args.destination, size=(args.size, args.size))