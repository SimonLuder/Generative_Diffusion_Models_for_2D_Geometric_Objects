
import json
import argparse

import torch
import clip

from utils.train_test_utils import initialize_model_weights, save_images_batch
from model.UNet import UNet
from model.ddpm import Diffusion as DDPMDiffusion


def inference(run_name, epoch, condition):

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
                    in_channel=args.image_channels,
                    out_channel=args.image_channels,
                    cfg_encoding=args.cfg_encoding,
                    num_classes=args.num_classes,
                    cfg_model_name=args.encoder_model,
                    device=args.device,
                    act=args.act,
                    ).to(args.device)
    else:
        print(f'"{args.model}" model not implemented')


    initialize_model_weights(model=model, weight_path=f"runs/{run_name}/models/{epoch}/model.pt", device=args.device)
    model.to(args.device).eval()

    # set diffusion model
    diffusion = DDPMDiffusion(img_size=args.image_size, 
                              img_channels=args.image_channels,
                              noise_schedule=args.noise_schedule, 
                              beta_start=args.beta_start, 
                              beta_end=args.beta_end,
                              s=args.s,
                              device=args.device,
                          )
    
    # apply clip tokenization
    if args.cfg_encoding == "clip_text":
        condition = clip.tokenize(condition)

    condition = condition.to(args.device)

    # predict
    images = diffusion.sample(model=model, condition=condition, cfg_scale=3)
    images = images.cpu()

    save_images_batch(images, filenames=["test_image.jpg"] * images.shape[0], save_dir="./")


if __name__ == "__main__":

    run_name = "2D_GeoShape_64_linear_clip_text_1705056262"
    epoch = "latest"
    condition = "triangle with radius 84 aspect ratio 1.7 rotation -24 degrees located at (76, 92)"
    inference(run_name, epoch, condition)

