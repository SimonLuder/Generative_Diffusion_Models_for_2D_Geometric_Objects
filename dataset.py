import torch
import numpy as np
import pandas as pd
from PIL import Image


class ImageSentenceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches of Images and Prompts.
    
    Attributes:
        transform (callable, optional): Optional transform to be applied on a sample.
        image_files (list): List of image file paths.
        captions (list): List of image captions.
    """
    def __init__(self, labels_path, transform=None, preprocess=None):
        self.transform = transform
        self.preprocess = preprocess
        df = pd.read_csv(labels_path)
        self.image_files = df["file"].tolist()
        self.captions = df["prompt"].tolist()
        
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        caption = self.captions[idx]
        image = Image.open(filename)

        if self.preprocess:
            caption = self.preprocess(caption)

        if self.transform:
            image = self.transform(image)

        return image, caption, filename


class ImageTabularDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches of Images and feature vectors.
    The feature vectors have been one-hot encoded for categorical values and column-wise normalized.
    
    Attributes:
        transform (callable, optional): Optional transform to be applied on a sample.
        image_files (list): List of image file paths.
        captions (list): feature vectors.
    """
    def __init__(self, labels_path, transform=None):
        self.transform = transform
        df = pd.read_csv(labels_path)
        self.image_files = df["file"].tolist()
        # captions
        captions = df[["radius", "x", "y", "rotation", "aspect_ratio", "shape_name"]]
        captions = pd.get_dummies(captions, columns = ['shape_name'], dtype=int, drop_first=False)
        captions=(captions - captions.mean()) / captions.std()
        self.captions = captions.values
    
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        caption = torch.Tensor(self.captions[idx])
        image = Image.open(filename)
        if self.transform:
            image = self.transform(image)
        return image, caption, filename
    
class ImageImageDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches of Images as tensor and Images raw.
    The feature vectors have been one-hot encoded for categorical values and column-wise normalized.
    
    Attributes:
        transform (callable, optional): Optional transform to be applied on a sample.
        image_files (list): List of image file paths.
        captions (list): raw image.
    """
    def __init__(self, labels_path, transform=None, preprocess=None):
        self.transform = transform
        self.preprocess = preprocess
        df = pd.read_csv(labels_path)
        self.image_files = df["file"].tolist()

       
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image = Image.open(filename)
        caption = image

        if self.preprocess:
            caption = self.preprocess(caption)

        if self.transform:
            image = self.transform(image)
            
        return image, caption, filename
    