import os
import cv2
import argparse
from tqdm import tqdm


def resize_images(source_folder, dest_folder, size=(128, 128)):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for filename in tqdm(os.listdir(source_folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(source_folder, filename))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(dest_folder, filename), img)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="./data/shapes/", help="Source directory")
    parser.add_argument('--destination', type=str, default=f"./data/shapes32/")
    parser.add_argument('--size', type=int, default=32, help="Image size for square proportions")

  
    args = parser.parse_args()
    resize_images(source_folder=args.source, dest_folder=args.destination, size=(args.size, args.size))