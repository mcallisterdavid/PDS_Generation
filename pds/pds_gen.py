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

from datetime import datetime


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
args = parser.parse_args()

init_image_fn = args.init_image_fn

pds = PDS(PDSConfig(
    sd_pretrained_model_or_path='stabilityai/stable-diffusion-2-1-base'
))

# Blur logic
# kernel_size = 10
# sigma = 3.2
# im = gaussian_blur(im, kernel_size, sigma)

if init_image_fn is not None:
    reference = torch.tensor(plt.imread(init_image_fn))[..., :3]
    reference = reference.permute(2, 0, 1)[None, ...]
    reference = reference.to(pds.unet.device)

    reference_latent = pds.encode_image(reference)
    im = reference_latent
else:
    im = torch.randn((1, 4, 64, 64), device=pds.unet.device)

save_dir = 'results/pds_gen_sde_edit/%s_lr%.3f_seed%d' % (args.prompt.replace(' ', '_'), args.lr, args.seed)
os.makedirs(save_dir, exist_ok=True)

seed_everything(args.seed)

def decode_latent(latent):
    latent = latent.detach().to(device)
    with torch.no_grad():
        rgb = pds.decode_latent(latent)
    rgb = rgb.float().cpu().permute(0, 2, 3, 1)
    rgb = rgb.permute(1, 0, 2, 3)
    rgb = rgb.flatten(start_dim=1, end_dim=2)
    return rgb


# PDS + SDEdit Generation
batch_size = 1

im.requires_grad_(True)
im.retain_grad()

im_optimizer = torch.optim.AdamW([im], lr=args.lr, betas=(0.9, 0.99), eps=1e-15)
decodes = []
decodes_src = []

for step in tqdm(range(args.n_steps)):
    im_optimizer.zero_grad()

    if init_image_fn is None:
        # A (generation)
        skip_percentage = min(step / 1400, 0.97)
    else:
        # B (projection to manifold)
        skip_percentage = 0.97

    with torch.no_grad():
        pds.config.guidance_scale = args.guidance_scale
        pds_dict = pds.pds_gen_sdedit_src(
            im=im,
            prompt=args.prompt,
            src_method="step",
            skip_percentage = skip_percentage,
            num_solve_steps = 12 + min(step // 200, 20),
            return_dict=True
        )
    grad = pds_dict['grad']
    src_x0 = pds_dict['src_x0']

    # loss.backward()
    im.backward(gradient=grad)
    im_optimizer.step()

    if step % 10 == 0:
        decoded = decode_latent(im.detach()).cpu().numpy()
        decodes.append(decoded)
        decoded_src = decode_latent(src_x0.detach()).cpu().numpy()
        decodes_src.append(decoded_src)
        plt.imsave(os.path.join(save_dir ,'pds_gen_sdedit_debug.png'), decoded)
        plt.imsave(os.path.join(save_dir ,'pds_gen_sdedit_x0_debug.png'), decoded_src)

    if step % 100 == 0:
        imageio.mimwrite(os.path.join(save_dir, 'pds_cat_tgt.mp4'), np.stack(decodes).astype(np.float32)*255, fps=10, codec='libx264')
        imageio.mimwrite(os.path.join(save_dir, 'pds_cat_x0.mp4'), np.stack(decodes_src).astype(np.float32)*255, fps=10, codec='libx264')