import argparse
import json
import os
from PIL import Image
from pathlib import Path


import torch

from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torchvision
from matplotlib import pyplot as plt
from model.ddpm import Diffusion as DDPMDiffusion


def generate_noise_steps():
    # Parameter settings
    parser = argparse.ArgumentParser()

    # Input images
    parser.add_argument("--images_path", type=str, default="./data/train64/")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--image_channels", type=int, default=1)

    parser.add_argument("--save_path", type=str, default="./images/noised_samples/")

    parser.add_argument("--batch_size", type=int, default=10)
    
    # Noise schedule
    # Options: linear, cosine
    parser.add_argument("--noise_schedule", type=str, default="cosine")
    # For linear noise schedule only
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    # For cosine noise schedule only
    parser.add_argument("--s", type=float, default=0.008)

    args = parser.parse_args()



    if args.image_channels == 3:
        transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    elif args.image_channels == 1:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5), (0.5))
            ])
   

    save_path = os.path.join(args.save_path)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving images at: {save_path}")

    dataset = torchvision.datasets.ImageFolder(root=args.images_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Recreate the folder
    os.makedirs(name=save_path, exist_ok=True)
    # Diffusion model initialization

    print("test")

    diffusion = DDPMDiffusion(img_size=args.image_size, 
                              img_channels=args.image_channels,
                              noise_schedule=args.noise_schedule, 
                              beta_start=args.beta_start, 
                              beta_end=args.beta_end,
                              s=args.s,
                              device="cpu",
                          )

    # Get image and noise tensor
    image = next(iter(dataloader))[0]
    print(image.shape)


    time = torch.Tensor([0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]).long()

    # Add noise to the image
    noised_image, _ = diffusion.noise_images(x=image, t=time)
    save_image(tensor=noised_image.add(1).mul(0.5), fp=os.path.join(save_path, f"{args.noise_schedule}.jpg"), nrow = len(time))

    print("success")


if __name__ == "__main__":
    generate_noise_steps()