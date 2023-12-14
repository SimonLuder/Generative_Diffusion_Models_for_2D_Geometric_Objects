import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch

from metrics import iou_pytorch, center_distance_pytorch, center_shapes
from utils.train_test_utils import get_dataloader, save_images_batch, initialize_model_weights, save_as_json
from model.UNet import UNet
from model.ddpm import Diffusion as DDPMDiffusion

def load_config(run_name):

    with open(f'runs/{run_name}/config.json') as json_file:
        config_dict = json.load(json_file)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    argparse_dict = vars(args)
    argparse_dict.update(config_dict)
    return args


def load_model(weight_path, args):

    if args.model == "unet_base":
        model = UNet(image_size=args.image_size, 
                    cfg_encoding=args.cfg_encoding,
                    num_classes=args.num_classes,
                    cfg_model_name=args.encoder_model,
                    device=args.device,
                    act=args.act)
    else:
        print(f'"{args.model}" model not implemented')

    initialize_model_weights(model=model, weight_path=weight_path, device=args.device)
    return model


def load_diffusion(args):
    # setup diffusion
    diffusion = DDPMDiffusion(img_size=args.image_size, 
                              img_channels=args.image_channels,
                              noise_schedule=args.noise_schedule, 
                              beta_start=args.beta_start, 
                              beta_end=args.beta_end,
                              s=args.s,
                              device=args.device,
                          )
    return diffusion


def eval_loop(dataloader, model, diffusion, save_dir, args):

    # metrics and log
    log_data={"samples":[]}
    total_iou = 0
    total_cdist = 0
    n = 0

    pbar = tqdm(dataloader)
    for (images, condition, filenames) in pbar:
        
        if args.cfg_encoding == "class":
            condition = torch.arange(0, args.num_classes, 1)

        images = images.to(args.device) 
        condition = condition.to(args.device)

        predicted_images = diffusion.sample(model, condition=condition) 

        imgs_norm_test = ((images.clamp(-1, 1) + 1) / 2).type(torch.float32) # images have range -1 to 1
        imgs_norm_pred = (predicted_images / 255).type(torch.float32) # predicted_images have range 0 to 255
        
        iou = iou_pytorch(imgs_norm_test, imgs_norm_pred).cpu()
        cdist = center_distance_pytorch(imgs_norm_test, imgs_norm_pred).cpu()
        iou_c = iou_pytorch(center_shapes(imgs_norm_test), 
                            center_shapes(imgs_norm_pred)).cpu()

        for ixd in range(images.shape[0]):
            sample = {"path_original":filenames[ixd],
                      "path_generated":os.path.join(save_dir, os.path.basename(filenames[ixd])),
                      "IoU":iou[ixd].item(), 
                      "IoU_centered":iou_c[ixd].item(),
                      "l2_distance":cdist[ixd].item(), 
                      }
            
            log_data["samples"].append(sample)
            
        save_images_batch(predicted_images, 
                          filenames=filenames, 
                          save_dir=save_dir,
                          )
        
        total_iou += torch.nansum(iou).item()
        total_cdist += torch.nansum(cdist).item()
        n += len(images)

    log_data["mean_iou"] = total_iou / n
    log_data["mean_cdist"] = total_cdist / n

    return log_data



if __name__ == "__main__":

    run_name = "2D_GeoShape_32_linear_clip_1701935944"
    epoch = "latest"

    args = load_config(run_name)
    
    # get device 
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataloader
    dataloader = get_dataloader(args=args)
    
    # unet model
    model = load_model(f"runs/{run_name}/models/{epoch}/model.pt", args=args)
    model.to(args.device).eval()

    # diffusion process
    diffusion = load_diffusion(args=args)

    save_dir = f"runs/{run_name}/test/{epoch}"
    Path( save_dir ).mkdir( parents=True, exist_ok=True )

    log_data = eval_loop(dataloader=dataloader["test"], 
                         model=model,
                         diffusion=diffusion,
                         save_dir=save_dir,
                         args=args)

    save_as_json(log_data, filepath=save_dir)
