import os
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

from diffusers import DDIMPipeline, DDIMScheduler, DDPMPipeline, DDPMScheduler, StableDiffusionPipeline
from typing import *
from jaxtyping import *

from pds import PDS, PDSConfig

parser = argparse.ArgumentParser()
parser.add_argument('--src_prompt', type=str, default='a pixelated, blurry, noisy, malformed, low-res image of a bedroom')
parser.add_argument('--tgt_prompt', type=str, default='a high-quality, high resolution, DSLR photo of a bedroom')
parser.add_argument('--src_image', type=str, default='./results/pds_gen_sde_edit/a_DSLR_photo_of_a_dog_in_a_winter_wonderland_lr0.100_seed0/pds_gen_sdedit_debug_bak.png')
parser.add_argument('--guidance_scale', type=float, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_steps', type=int, default=2000)
args = parser.parse_args()

pds = PDS(PDSConfig(
    sd_pretrained_model_or_path='stabilityai/stable-diffusion-2-1-base'
))

reference = torch.tensor(plt.imread(args.src_image))
reference = reference[..., :3].permute(2, 0, 1)[None, ...]
reference = reference.to(pds.unet.device)

save_dir = os.path.dirname(args.src_image)

def decode_latent(latent):
    latent = latent.detach().to(device)
    with torch.no_grad():
        rgb = pds.decode_latent(latent)
    rgb = rgb.float().cpu().permute(0, 2, 3, 1)
    rgb = rgb.permute(1, 0, 2, 3)
    rgb = rgb.flatten(start_dim=1, end_dim=2)
    return rgb

reference_latent = pds.encode_image(reference)
decoded = decode_latent(reference_latent)

im = reference_latent.clone().to(device)
im.requires_grad_(True)
im.retain_grad()

im_optimizer = torch.optim.AdamW([im], lr=0.01, betas=(0.9, 0.99), eps=1e-15)

decodes = []
for step in tqdm(range(args.n_steps)):
    im_optimizer.zero_grad()

    with torch.no_grad():
        pds_dict = pds(
            tgt_x0=im,
            src_x0=reference_latent.clone(), # Normal PDS
            # src_x0=im.clone(), # src and tgt are same
            tgt_prompt=args.tgt_prompt,
            src_prompt=args.src_prompt,
            return_dict=True
        )
    grad = pds_dict['grad']

    im.backward(gradient=grad)
    im_optimizer.step()

    if step % 10 == 0:
        decoded = decode_latent(im.detach()).cpu().numpy()
        decodes.append(decoded)
        plt.imsave(os.path.join(save_dir ,'pds_gen_sdedit_postprocess_debug.png'), decoded)

    if step % 100 == 0:
        imageio.mimwrite(os.path.join(save_dir, 'pds_cat_tgt_edit.mp4'), np.stack(decodes).astype(np.float32)*255, fps=10, codec='libx264')