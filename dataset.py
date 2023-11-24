import torch
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
    def __init__(self, labels_path, transform=None):
        self.transform = transform
        df = pd.read_csv(labels_path)
        self.image_files = df["file"].tolist()
        self.captions = df["prompt"].tolist()
        
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        caption = self.captions[idx]
        image = Image.open(filename)
        if self.transform:
            image = self.transform(image)
        return image, caption, filename
    