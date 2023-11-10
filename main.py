import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim


# from torch.utils.tensorboard import SummaryWriter

from model.UNet import UNet
from model.ddpm import Diffusion as DDPMDiffusion
from train_utils import setup_logging, get_data, save_images, initialize_model_weights


def main():

    # setup logging
    setup_logging(args.run_name)

    # get device 
    device = args.device
    
    # setup config

    # load model
    model = UNet(image_size=args.image_size, 
                 num_classes=args.num_classes,
                 device=device,
                 act=args.act,
                 ).to(device)
    
    print(f'Parameters (total): {sum(p.numel() for p in model.parameters()):_d}')
    print(f'Parameters (train): {sum(p.numel() for p in model.parameters() if p.requires_grad):_d}')


    # Model optimizer
    if args.optim == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    elif args.optim == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr)
    else:
        raise ValueError(f'"{args.optim}" optimizer not implemented')
    
    # Resume Training
    if args.resume == True:
        from_epoch = args.from_epoch - 1
        model_path = os.path.join(args.load_model_dir, f"model_{from_epoch}.pt" )
        initialize_model_weights(model=model, weight_path=model_path, device=args.device)
        
        optim_path = os.path.join(args.load_model_dir, f"optim_{from_epoch}.pt" )
        optim_weight_dict = torch.load(f=optim_path, map_location=device)
        optimizer.load_state_dict(state_dict=optim_weight_dict)
    else:
        from_epoch = 0

    # Dataloader
    dataloader = get_data(args)

    mse = nn.MSELoss()

    diffusion = DDPMDiffusion(img_size=args.image_size, 
                              noise_schedule=args.noise_schedule, 
                              beta_start=args.beta_start, 
                              beta_end=args.beta_end,
                              s=args.s,
                              device=device,
                          )
    
    # logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(from_epoch, args.epochs):
        print(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, condition) in enumerate(pbar):
            images = images.to(device)
            condition = condition.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            # set propoportion of conditions to zero
            if np.random.random() < 0.1:
                contidion = None

            predicted_noise = model(x=x_t, time=t, y=condition)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            # logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))

        # save latest model
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"model_latest.pt"))
        torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim_latest.pt"))

        # save model periodically
        if args.create_checkpoints and epoch % args.checkpoints_interval == 0:
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"model_{epoch}.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim_{epoch}.pt"))




if __name__ == "__main__":

    # config
    parser = argparse.ArgumentParser()

    # ===========================================Base Settings============================================
    # Total epochs for training
    parser.add_argument("--epochs", type=int, default=500)
    # Batch size for training
    parser.add_argument("--batch_size", type=int, default=8)
    # Dataset path
    # Set path to folder where training images are located
    parser.add_argument("--dataset_path", type=str, default="./data2/")
    # Input image size
    parser.add_argument("--image_size", type=int, default=32)

    # Set noise schedule for the model
    # Options: linear, cosine
    parser.add_argument("--noise_schedule", type=str, default="cosine")
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    # for cosine noise schedule only
    parser.add_argument("--s", type=float, default=0.008)

    # Set optimizer for the model
    # Options: adam, adamw
    parser.add_argument("--optim", type=str, default="adamw")
    # Set activation function used in the UNet
    # Options: relu, relu6, silu, lrelu, gelu
    parser.add_argument("--act", type=str, default="silu")
    # Set device for training
    # Options: cpu or cuda
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'))

    # If conditional generation is trained 
    parser.add_argument("--num_classes", type=int, default=None)

    # ===========================================Training Checkpoints======================================
    # Set if model checkpoints are saved
    parser.add_argument("--create_checkpoints", type=bool, default=True)
    # Set per how many epochs a model checkpoint is created
    parser.add_argument("--checkpoints_interval", type=int, default=50)

    # ===========================================Resume Training===========================================
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--from_epoch", type=int, default=-1)
    parser.add_argument("--load_model_dir", type=str, default="")

    args = parser.parse_args()

    args.run_name = f"2d_geometric_shapes_{args.image_size}_{np.round(time.time(), 0)}"

    args.device = "cuda"
    args.lr = 3e-4


    main()