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

import clip

from model.UNet import UNet
from model.ddpm import Diffusion as DDPMDiffusion
from utils.wandb import WandbManager, WandbTable, wandb_image
from utils.train_utils import setup, get_dataloader, save_images_batch, initialize_model_weights
from metrics import iou_pytorch, center_distance_pytorch


def main():

    # Setup WandbManager
    wandb_manager = WandbManager(vars(args))
    # init run
    run = wandb_manager.init()

    # setup logging
    setup(args)

    # get device 
    device = args.device
    
    # load model
    model = UNet(image_size=args.image_size, 
                 cfg_encoding=args.cfg_encoding,
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
    dataloader = get_dataloader(args)

    mse = nn.MSELoss()

    diffusion = DDPMDiffusion(img_size=args.image_size, 
                              noise_schedule=args.noise_schedule, 
                              beta_start=args.beta_start, 
                              beta_end=args.beta_end,
                              s=args.s,
                              device=device,
                          )
    
    for epoch in range(from_epoch, args.epochs):
        print(f"Starting epoch {epoch}:")

        log_data=dict()
        log_data["epoch"] = epoch
        
        epoch_loss = 0
        pbar = tqdm(dataloader["train"])
        for i, (images, condition, _) in enumerate(pbar):
            
            # apply clip tokenization
            if args.cfg_encoding == "clip":
                condition = clip.tokenize(condition)

            images = images.to(device)
            condition = condition.to(device)

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            # set propoportion of conditions to zero
            if args.num_classes is None or np.random.random() < 0.1:
                condition = None

            predicted_noise = model(x=x_t, time=t, y=condition)
            loss = mse(noise, predicted_noise)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())

        log_data["epoch_loss"] = epoch_loss

        # validation loop
        if epoch % args.val_interval == 0:
    
            save_dir = f"results/{args.run_name}/{epoch}"
            Path( save_dir ).mkdir( parents=True, exist_ok=True )

            wandb_log_table = WandbTable()

            total_iou = 0
            total_cdist = 0
            n = 0

            pbar = tqdm(dataloader["val"])
            for (images, condition, filenames) in pbar:

                raw_condition = condition
                
                if args.cfg_encoding == "class":
                    condition = torch.arange(0, args.num_classes, 1)

                # apply clip tokenization
                if args.cfg_encoding == "clip":
                    condition = clip.tokenize(condition)

                images = images.to(device)
                condition = condition.to(device)

                predicted_images = diffusion.sample(model, condition=condition)

                iou = iou_pytorch(images, predicted_images).cpu()
                cdist = center_distance_pytorch(images, predicted_images).cpu()

                for ixd in range(images.shape[0]):
                    result= {"true_image":wandb_image(images[ixd].cpu().numpy()), 
                            "generated_image":wandb_image(predicted_images[ixd].cpu().numpy()),
                            "condition":raw_condition[ixd],
                            "IoU":iou[ixd].item(), 
                            "l2_distance":cdist[ixd].item(), 
                            "epoch":epoch,
                            "filename":filenames[ixd]
                            }
                    wandb_log_table.add_data(result)

                # save_images_batch(predicted_images, 
                #                   filenames=filenames, 
                #                   save_dir=save_dir,
                #                   )
                
                total_iou += torch.nansum(iou).item()
                total_cdist += torch.nansum(cdist).item()
                n += len(images)

 
            run.log(data={"validation_results": wandb_log_table.get_table()})

            log_data["mean_iou"] = total_iou / n
            log_data["mean_cdist"] = total_cdist / n
   
                
        run.log(data=log_data, step=epoch)

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

    # ===========================================Run Settings=============================================
    parser.add_argument("--project", type=str, default=f"MSE_P7")
    parser.add_argument("--run_name", type=str, default=f"2D_GeoShape_32_linear_clip_{time.time():.0f}")


    # ===========================================Base Settings============================================
    # Total epochs for training
    parser.add_argument("--epochs", type=int, default=500)
    # Batch size for training
    parser.add_argument("--batch_size", type=int, default=8)
    # Input image size
    parser.add_argument("--image_size", type=int, default=32)
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
    parser.add_argument("--val_images", type=str, default="./data/test32/images/")

    # Only relevant if cfg_encoding is "clip"
    parser.add_argument("--train_labels", type=str, default="data/train32/labels_20.csv")
    parser.add_argument("--val_labels", type=str, default="data/test32/labels_2.csv")

    # ===========================================Validation===============================================
    parser.add_argument("--val_interval", type=int, default=50)

    # ===========================================Conditioning=============================================
    # If conditional generation is trained 
    parser.add_argument("--cfg_encoding", type=str, default="clip")
    # only relevant if cfg_encoding is set to "classes"
    parser.add_argument("--num_classes", type=int, default=9)
  

    # ===========================================Training Checkpoints======================================
    # Set if model checkpoints are saved
    parser.add_argument("--create_checkpoints", type=bool, default=True)
    # Set per how many epochs a model checkpoint is created
    parser.add_argument("--checkpoints_interval", type=int, default=100)


    # ===========================================Resume Training===========================================
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--from_epoch", type=int, default=-1)
    parser.add_argument("--load_model_dir", type=str, default="")

    args = parser.parse_args()

    main()