import os
import random
import argparse
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
device = torch.device('cuda')
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import mediapy
import imageio
from utils.imageutil import gaussian_blur, gaussian_kernel

from diffusers import DDIMPipeline, DDIMScheduler, DDPMPipeline, DDPMScheduler, StableDiffusionPipeline
from typing import *
from jaxtyping import *

from pds import PDS, PDSConfig
from pds_sdxl import PDS_sdxl, PDS_sdxlConfig

from datetime import datetime
import matplotlib.pyplot as plt


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, default='a DSLR photo of a dog in a winter wonderland')
parser.add_argument('--init_image_fn', type=str, default=None)
parser.add_argument('--skip_percentage', type=float, default=0.8)
parser.add_argument('--num_solve_steps', type=int, default=32)
parser.add_argument('--guidance_scale', type=float, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_steps', type=int, default=3000)
parser.add_argument('--model', type=str, default='sd', choices=['sd', 'sdxl'])
args = parser.parse_args()

init_image_fn = args.init_image_fn

if args.model == 'sd':
    pds = PDS(PDSConfig(
        sd_pretrained_model_or_path='stabilityai/stable-diffusion-2-1-base'
    ))
    latent_dim = 64
else:
    pds = PDS_sdxl(PDS_sdxlConfig(
        sd_pretrained_model_or_path="stabilityai/stable-diffusion-xl-base-1.0"
    ))
    latent_dim = 128


if init_image_fn is not None:
    reference = torch.tensor(plt.imread(init_image_fn))[..., :3]
    reference = reference.permute(2, 0, 1)[None, ...]
    reference = reference.to(pds.unet.device)

    reference_latent = pds.encode_image(reference)
    im = reference_latent.clone()
else:
    im = torch.randn((1, 4, latent_dim, latent_dim), device=pds.unet.device)

# save_dir = 'results/%s_pds_gen_sde_edit/%s_lr%.3f_seed%d' % (args.model, args.prompt.replace(' ', '_'), args.lr, args.seed)
# os.makedirs(save_dir, exist_ok=True)

seed_everything(args.seed)

def decode_latent(latent):
    latent = latent.detach().to(device)
    with torch.no_grad():
        rgb = pds.decode_latent(latent)
    rgb = rgb.float().cpu().permute(0, 2, 3, 1)
    rgb = rgb.permute(1, 0, 2, 3)
    rgb = rgb.flatten(start_dim=1, end_dim=2)
    return rgb

# pds.scheduler.set_timesteps(pds.config.num_inference_steps)
pds.scheduler.set_timesteps(pds.config.num_inference_steps)
timesteps = reversed(pds.scheduler.timesteps)

min_step = 1 if pds.config.min_step_ratio <= 0 else int(len(timesteps) * pds.config.min_step_ratio)
max_step = (
    len(timesteps) if pds.config.max_step_ratio >= 1 else int(len(timesteps) * pds.config.max_step_ratio)
)
max_step = max(max_step, min_step + 1)

x0_coeffs = []
eps_coeffs = []
for step in range(min_step, max_step):
    t = timesteps[step]
    t_prev = timesteps[step - 1]
    beta_t = pds.scheduler.betas[t]
    alpha_t = pds.scheduler.alphas[t]
    alpha_bar_t = pds.scheduler.alphas_cumprod[t]
    alpha_bar_t_prev = pds.scheduler.alphas_cumprod[t_prev]
    gamma_t = torch.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)
    delta_t = torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
    sigma_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t) ** (0.5)
    x0_coeff = (torch.sqrt(alpha_bar_t_prev) - gamma_t - delta_t*torch.sqrt(alpha_bar_t)) / sigma_t
    eps_coeff = gamma_t * torch.sqrt(1/alpha_bar_t-1) / sigma_t
    x0_coeffs.append(x0_coeff.item())
    eps_coeffs.append(eps_coeff.item())

plt.figure(figsize=(8, 5))
steps = [t*2 for t in range(min_step, max_step)]
plt.plot(steps, x0_coeffs, label='x0')
plt.plot(steps, eps_coeffs, label='eps')
# set x axis to log scale
# plt.xscale('log')
plt.legend()
plt.xlabel('time steps')
plt.savefig('vis.jpg')