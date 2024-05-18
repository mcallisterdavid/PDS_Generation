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
parser.add_argument('--src_method', type=str, default='sdedit', choices=['sdedit', 'step'])
args = parser.parse_args()

init_image_fn = args.init_image_fn
src_method = args.src_method

if args.model == 'sd':
    pds = PDS(PDSConfig(
        sd_pretrained_model_or_path='stabilityai/stable-diffusion-2-1-base',
        texture_inversion_embedding='/nfshomes/songweig/hcl/code/PDS_Generation/pds/assets/-steps-6000.bin'
    ))
    latent_dim = 64
else:
    pds = PDS_sdxl(PDS_sdxlConfig(
        sd_pretrained_model_or_path="stabilityai/stable-diffusion-xl-base-1.0"
    ))
    latent_dim = 128


pds.config.num_inference_steps = args.n_steps
# Blur logic
# kernel_size = 10
# sigma = 3.2
# im = gaussian_blur(im, kernel_size, sigma)

if init_image_fn is not None:
    reference = torch.tensor(plt.imread(init_image_fn))[..., :3]
    reference = reference.permute(2, 0, 1)[None, ...]
    reference = reference.to(pds.unet.device)

    reference_latent = pds.encode_image(reference)
    im = reference_latent.clone()
else:
    im = torch.randn((1, 4, latent_dim, latent_dim), device=pds.unet.device) * 0.1

save_dir = 'results/%s_reverse_process_visualization/%s_nstep%d_seed%d' % (args.model, args.prompt.replace(' ', '_'), args.n_steps, args.seed)
os.makedirs(save_dir, exist_ok=True)
print('Results will be saved to:', save_dir)

seed_everything(args.seed)

def decode_latent(latent):
    latent = latent.detach().to(device)
    with torch.no_grad():
        rgb = pds.decode_latent(latent)
    rgb = rgb.float().cpu().permute(0, 2, 3, 1)
    rgb = rgb.permute(1, 0, 2, 3)
    rgb = rgb.flatten(start_dim=1, end_dim=2)
    return rgb

# Returns timestep normalized to range (0, 1)
def sample_timestep(lower_bound, upper_bound):

    random_float = torch.rand(size=(1,)).item()
    return lower_bound + (upper_bound - lower_bound) * random_float

# PDS + SDEdit Generation
batch_size = 1  

im.requires_grad_(True)
im.retain_grad()

# im_optimizer = torch.optim.AdamW([im], lr=args.lr, betas=(0.9, 0.99), eps=1e-15)
im_optimizer = torch.optim.SGD([im], lr=args.lr)
decodes = []
decodes_src = []
latents_noisy_prev = None
noise_pred_prev = None
noise_pred = None
prompt=args.prompt

for step in tqdm(range(args.n_steps)):
    im_optimizer.zero_grad()

    with torch.no_grad():
        pds.config.guidance_scale = args.guidance_scale

        t_sample = (1 - step / args.n_steps) * 0.97 + 0.02

        latents_noisy=latents_noisy_prev
        noise=noise_pred_prev

        device = pds.device
        scheduler = pds.scheduler

        # process text.
        pds.update_text_features(tgt_prompt=prompt, src_prompt=prompt+', oversaturated, smooth, pixelated, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed')
        tgt_text_embedding = pds.tgt_text_feature
        src_text_embedding = pds.src_text_feature
        uncond_embedding = pds.null_text_feature

        batch_size = im.shape[0]
        t, t_prev = pds.pds_timestep_sampling(batch_size, sample=t_sample)

        if noise is None:
            noise = torch.randn_like(im)

        if latents_noisy is None:
            latents_noisy = scheduler.add_noise(im, noise, t)

        latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
        text_embeddings = torch.cat([tgt_text_embedding, uncond_embedding], dim=0)
        noise_pred = pds.unet.forward(
            latent_model_input,
            torch.cat([t] * 2).to(device),
            encoder_hidden_states=text_embeddings,
        ).sample
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred_hight = noise_pred_uncond + pds.config.guidance_scale * (noise_pred_text - noise_pred_uncond)

        x_0_hat_pred = scheduler.step(noise_pred_hight, t, latents_noisy).pred_original_sample

        # x_{t-1}, prompt
        latents_noisy_prev = scheduler.step(noise_pred_hight, t, latents_noisy, eta=0).prev_sample
        latent_model_input = torch.cat([latents_noisy_prev] * 2, dim=0)
        text_embeddings = torch.cat([tgt_text_embedding, uncond_embedding], dim=0)
        noise_pred_prev = pds.unet.forward(
            latent_model_input,
            torch.cat([t_prev] * 2).to(device),
            encoder_hidden_states=text_embeddings,
        ).sample
        noise_pred_text, noise_pred_uncond = noise_pred_prev.chunk(2)
        noise_pred_lowt = noise_pred_uncond + pds.config.guidance_scale * (noise_pred_text - noise_pred_uncond)

        # x_{t-1}, uncertain prompt
        text_embeddings = torch.cat([src_text_embedding, uncond_embedding], dim=0)
        noise_pred_prev = pds.unet.forward(
            latent_model_input,
            torch.cat([t_prev] * 2).to(device),
            encoder_hidden_states=text_embeddings,
        ).sample
        noise_pred_text, noise_pred_uncond = noise_pred_prev.chunk(2)
        noise_pred_lowt_yu = noise_pred_uncond + pds.config.guidance_scale * (noise_pred_text - noise_pred_uncond)

        # x_{t-1}, uncertain token
        pds.update_text_features(tgt_prompt=prompt, src_prompt=prompt+', <pds>')
        tgt_text_embedding = pds.tgt_text_feature
        src_text_embedding = pds.src_text_feature
        uncond_embedding = pds.null_text_feature
        text_embeddings = torch.cat([src_text_embedding, uncond_embedding], dim=0)
        noise_pred_prev = pds.unet.forward(
            latent_model_input,
            torch.cat([t_prev] * 2).to(device),
            encoder_hidden_states=text_embeddings,
        ).sample
        noise_pred_text, noise_pred_uncond = noise_pred_prev.chunk(2)
        noise_pred_lowt_yt = noise_pred_uncond + pds.config.guidance_scale * (noise_pred_text - noise_pred_uncond)


        x_0_hat_pred_hight = scheduler.step(noise_pred_hight, t, latents_noisy).pred_original_sample
        x_0_hat_pred_lowt = scheduler.step(noise_pred_lowt, t, latents_noisy).pred_original_sample
        x_0_hat_pred_lowt_yu = scheduler.step(noise_pred_lowt_yu, t, latents_noisy).pred_original_sample
        x_0_hat_pred_lowt_yt = scheduler.step(noise_pred_lowt_yt, t, latents_noisy).pred_original_sample

        alpha_prod_t_prev = scheduler.alphas_cumprod[t_prev].to(device)
        w = ((1 - alpha_prod_t_prev) / alpha_prod_t_prev) ** 0.5

        grad = w * (noise_pred_lowt - noise_pred_hight)
        grad = torch.nan_to_num(grad)

    noise_pred = noise_pred_hight
    noise_pred_prev = noise_pred_lowt
    latents_noisy_prev = latents_noisy_prev
    if step == 0:
        im = x_0_hat_pred

    # loss.backward()
    im = im - args.lr * grad

    decoded_hat_pred_hight = decode_latent(x_0_hat_pred_hight.detach()).cpu().numpy()
    plt.imsave(os.path.join(save_dir ,'x_0_hat_pred_hight_%d.png' % t), decoded_hat_pred_hight)

    decoded_0_hat_pred_lowt = decode_latent(x_0_hat_pred_lowt.detach()).cpu().numpy()
    plt.imsave(os.path.join(save_dir ,'x_0_hat_pred_lowt_%d.png' % t), decoded_0_hat_pred_lowt)

    decoded_hat_pred_lowt_yu = decode_latent(x_0_hat_pred_lowt_yu.detach()).cpu().numpy()
    plt.imsave(os.path.join(save_dir ,'x_0_hat_pred_lowt_yu_%d.png' % t), decoded_hat_pred_lowt_yu)

    decoded_hat_pred_lowt_yt = decode_latent(x_0_hat_pred_lowt_yt.detach()).cpu().numpy()
    plt.imsave(os.path.join(save_dir ,'x_0_hat_pred_lowt_yt_%d.png' % t), decoded_hat_pred_lowt_yt)

    decoded_cat = np.concatenate([decoded_hat_pred_hight, decoded_0_hat_pred_lowt, decoded_hat_pred_lowt_yt, decoded_hat_pred_lowt_yu], axis=1)
    plt.imsave(os.path.join(save_dir ,'x_0_hat_pred_cat_%d.png' % t), decoded_cat)


