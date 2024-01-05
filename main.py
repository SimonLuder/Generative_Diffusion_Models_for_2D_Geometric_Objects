import os
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim

from model.UNet import UNet
from model.ddpm import Diffusion as DDPMDiffusion
from utils.wandb import WandbManager, wandb_image
from utils.train_test_utils import get_dataloader, initialize_model_weights, save_as_json
from test import eval_loop


def main():
    print(f"Starting run: {args.run_name}")
    # save config from this run in 
    save_as_json(vars(args), f"runs/{args.run_name}/config.json")

    # local logging
    local_logs = list()

    # Setup WandbManager
    wandb_manager = WandbManager(vars(args))
    # init run
    run = wandb_manager.get_run()

    # get device 
    device = args.device
    
    # load model
    if args.model == "unet_base":
        model = UNet(image_size=args.image_size, 
                    in_channel=args.image_channels,
                    out_channel=args.image_channels,
                    cfg_encoding=args.cfg_encoding,
                    num_classes=args.num_classes,
                    cfg_model_name=args.encoder_model,
                    device=args.device,
                    act=args.act,
                    ).to(device)
    else:
        print(f'"{args.model}" model not implemented')
    
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
        optim_weight_dict = torch.load(f=optim_path, map_location=args.device)
        optimizer.load_state_dict(state_dict=optim_weight_dict)
    else:
        from_epoch = 0

    # Dataloader
    dataloader = get_dataloader(args)

    mse = nn.MSELoss()
    

    diffusion = DDPMDiffusion(img_size=args.image_size, 
                              img_channels=args.image_channels,
                              noise_schedule=args.noise_schedule, 
                              beta_start=args.beta_start, 
                              beta_end=args.beta_end,
                              s=args.s,
                              device=args.device,
                          )
    
    for epoch in range(from_epoch, args.epochs):
        print(f"Starting epoch {epoch}:")

        log_data={"train": dict()}
        log_data["train"]["epoch"] = epoch
        
        epoch_loss = 0
        pbar = tqdm(dataloader["train"])
        for i, (images, condition, _) in enumerate(pbar):

            images = images.to(device)
            condition = condition.to(device)

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            # set propoportion of conditions to zero
            if args.cfg_encoding is None or np.random.random() < 0.1:
                condition = None

            predicted_noise = model(x=x_t, time=t, y=condition)
            loss = mse(noise, predicted_noise)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())

        log_data["train"]["epoch_loss"] = epoch_loss
        log_data["train"]["time"] = time.time()

        # validation loop
        if args.val_interval > 0 and ((epoch % args.val_interval == 0) or (epoch + 1 == args.epochs)):

            model.eval()
    
            save_dir = f"runs/{args.run_name}/validation/{epoch}"
            Path( save_dir ).mkdir( parents=True, exist_ok=True )

            log_data["val"] = eval_loop(dataloader=dataloader["val"], 
                                        model=model, 
                                        diffusion=diffusion, 
                                        save_dir=save_dir,
                                        args=args,
                                        )
            log_data["val"]["epoch"] = epoch
            log_data["val"]["time"] = time.time()

            # log to wandb
            df_samples = pd.DataFrame.from_dict(log_data["val"]["samples"])
            df_samples["img_original"] = df_samples["path_original"].apply(lambda x: wandb_image(path = x))
            df_samples["img_generated"] = df_samples["path_generated"].apply(lambda x: wandb_image(path = x))
            df_samples["epoch"] = epoch
            wandb_manager.log_dataframe( "validation_samples", df_samples)

            model.train()
   
        
        run.log(data=log_data)
        local_logs.append(log_data)
        save_as_json(local_logs, f"runs/{args.run_name}/metrics.json")


        # save latest model
        model_path = os.path.join("runs/", args.run_name, "models", "latest",  "model.pt")
        # optim_path = os.path.join("runs/", args.run_name, "models", "latest", "optim.pt")
        Path(os.path.split(model_path)[0]).mkdir( parents=True, exist_ok=True )
        # Path(os.path.split(optim_path)[0]).mkdir( parents=True, exist_ok=True )
        torch.save(model.state_dict(), model_path)
        # torch.save(optimizer.state_dict(), optim_path)

        # save model periodically
        if args.create_checkpoints and ((epoch % args.checkpoints_interval == 0) or (epoch + 1 == args.epochs)):
            model_path = os.path.join("runs/", args.run_name, "models", str(epoch),  "model.pt")
            # optim_path = os.path.join("runs/", args.run_name, "models", str(epoch), "optim.pt")
            Path(os.path.split(model_path)[0]).mkdir( parents=True, exist_ok=True )
            # Path(os.path.split(optim_path)[0]).mkdir( parents=True, exist_ok=True )
            torch.save(model.state_dict(), model_path)
            # torch.save(optimizer.state_dict(), optim_path)

    wandb_manager.log_everything(run_name=args.run_name, path=f"runs/{args.run_name}")
            
    

if __name__ == "__main__":

    # config
    parser = argparse.ArgumentParser()
    # ===========================================Run Settings=============================================
    parser.add_argument("--project", type=str, default=f"MSE_P7")
    parser.add_argument("--run_name", type=str, default=None)

    # ===========================================Base Settings============================================
    parser.add_argument("--model", type=str, default="unet_base")
    # Total epochs for training
    parser.add_argument("--epochs", type=int, default=10)
    # Batch size for training
    parser.add_argument("--batch_size", type=int, default=12)
    # Input image size
    parser.add_argument("--image_size", type=int, default=32)
    # Image channels
    parser.add_argument("--image_channels", type=int, default=1)
    # Set noise schedule
    # Options: linear, cosine
    parser.add_argument("--noise_schedule", type=str, default="linear")
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    # For cosine noise schedule only
    parser.add_argument("--s", type=float, default=0.008)

    # Set optimizer for the model
    # Options: adam, adamw
    parser.add_argument("--optim", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=3e-4)

    # Set activation function used in the UNet
    # Options: relu, relu6, silu, lrelu, gelu
    parser.add_argument("--act", type=str, default="silu")
    # Set device for training
    # Options: cpu or cuda
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'))

    # ===========================================Dataset==================================================
    # Set path to folder where images are located
    parser.add_argument("--train_images", type=str, default="./data/train32/images/")
    parser.add_argument("--val_images", type=str, default="./data/val32/images/")
    parser.add_argument("--test_images", type=str, default="./data/test32/images/")

    # Set path to file with image labels
    parser.add_argument("--train_labels", type=str, default="./data/train32/labels.csv")
    parser.add_argument("--val_labels", type=str, default="./data/val32/labels.csv")
    parser.add_argument("--test_labels", type=str, default="./data/test32/labels.csv")

    # ===========================================Validation===============================================
    parser.add_argument("--val_interval", type=int, default=25)

    # ===========================================Conditioning=============================================
    # If conditional generation is trained 
    parser.add_argument("--cfg_encoding", type=str, default="cnn_image") # tabular, clip_text, clip_image, cnn_image
    # only relevant if cfg_encoding is set to "clip_image" or "clip_text"
    parser.add_argument("--encoder_model", type=str, default="ViT-B/32")
    # only relevant if cfg_encoding is set to "classes"
    parser.add_argument("--num_classes", type=int, default=None)
  

    # ===========================================Training Checkpoints======================================
    # Set if model checkpoints are saved
    parser.add_argument("--create_checkpoints", type=bool, default=True)
    # Set per how many epochs a model checkpoint is created
    parser.add_argument("--checkpoints_interval", type=int, default=25)

    # ===========================================Resume Training===========================================
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--from_epoch", type=int, default=-1)
    parser.add_argument("--load_model_dir", type=str, default="")

    args = parser.parse_args()

    # set default run name if nothing else is specified
    if args.run_name is None:
        args.run_name = f"2D_GeoShape_{args.image_size}_{args.noise_schedule}_{args.cfg_encoding}_{time.time():.0f}"

    main()