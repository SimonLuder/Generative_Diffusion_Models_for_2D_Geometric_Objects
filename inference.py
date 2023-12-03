
import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import cv2

import torch

from metrics import iou_pytorch, center_distance_pytorch
from utils.train_test_utils import get_dataloader, save_images_batch, initialize_model_weights, save_as_json
from utils.wandb import WandbManager, wandb_image
from model.UNet import UNet
from model.ddpm import Diffusion as DDPMDiffusion


def inference(run_name, epoch):

    # open config
    with open(f'runs/{run_name}/config.json') as json_file:
        config_dict = json.load(json_file)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    argparse_dict = vars(args)
    argparse_dict.update(config_dict)

    # set device 
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    if args.model == "unet_base":
        model = UNet(image_size=args.image_size, 
                    cfg_encoding=args.cfg_encoding,
                    num_classes=args.num_classes,
                    device=args.device,
                    act=args.act)
    else:
        print(f'"{args.model}" model not implemented')

    initialize_model_weights(model=model, weight_path=f"runs/{run_name}/models/{epoch}/model.pt", device=args.device)
    model.eval()

    # set diffusion model
    diffusion = DDPMDiffusion(img_size=args.image_size, 
                              noise_schedule=args.noise_schedule, 
                              beta_start=args.beta_start, 
                              beta_end=args.beta_end,
                              s=args.s,
                              device=args.device,
                          )
    
    # predict
    image = diffusion.sample(model=model, condition="test")
    cv2.imshow(image.cpu().numpy())


if __name__ == "__main__":

    run_name = "2D_GeoShape_32_linear_clip_1701532567"
    epoch = "latest"
    inference(run_name, epoch)
