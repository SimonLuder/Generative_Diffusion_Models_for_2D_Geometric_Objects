import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim


class Diffusion:
    def __init__(self, noise_schedule="linear", noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_schedule = noise_schedule
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # Noise steps
        self.beta = self.prepare_noise_schedule().to(device)
        # Formula: α = 1 - β
        self.alpha = 1. - self.beta
        # The cumulative sum of α.
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self, s=0.008):
        if self.noise_schedule == "linear":
            # simple linear noise schedule
            beta_t =  torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif self.noise_schedule == "cosine":
            # create cosine noise schedule
            steps = self.noise_steps + 1
            t = torch.linspace(0, self.noise_steps, steps)
            alpha_t = torch.cos(((t / self.noise_steps + s) / (1 + s)) * (np.pi / 2)).pow(2)
            alpha_t = alpha_t / alpha_t[0]
            # caluculate betas
            beta_t = 1 - alpha_t[1:] / alpha_t[:-1]
            # clip beta_t at 0.999 to pretenv singularities
            beta_t = beta_t.clamp(max=0.999)
        else:
            raise NotImplementedError(f"'{self.noise_schedule}' schedule is no implemented!")
        
        return beta_t

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        print(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

