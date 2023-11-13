import os
import torch
import torchvision
import clip
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def resume_from_checkpoint():
    raise NotImplementedError(f"'resume_from_checkpoint' is no implemented!")
 

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size 80 for args.image_size = 64
            torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    if args.cfg_encoding == None or args.cfg_encoding == "classes":
        print("classes loading")
        dataset = torchvision.datasets.ImageFolder(root=args.dataset_path, transform=transforms)

    if args.cfg_encoding == "clip":
        print("captions loading")
        dataset = ImageSentenceDataset(args.dataset_path, captions_file=args.captions_file, transform=transforms)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader
    

def sample_condition(args):
    condition = None
    if args.cfg_encoding is None:
        pass

    elif args.cfg_encoding == "class" and args.num_classes is not None:
        condition = torch.arange(0, args.num_classes, 1)

    elif args.cfg_encoding == "clip" and args.val_caption_file is not None:
        df = pd.read_csv(args.val_caption_file, header=None, names=["image_files", "captions"])
        condition = clip.tokenize(df["captions"].tolist())

    return condition


class ImageSentenceDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, captions_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        df = pd.read_csv(captions_file, header=None, names=["image_files", "captions"])
        self.image_files = df["image_files"].tolist()
        self.captions = df["captions"].tolist()
        
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        caption = self.captions[idx]
        image = Image.open(os.path.join(self.root_dir, filename))
        if self.transform:
            image = self.transform(image)
        return image, caption
    
    
def initialize_model_weights(model, weight_path, device):
    model_weights_dict = torch.load(f=weight_path, map_location=device)
    model.load_state_dict(model_weights_dict)