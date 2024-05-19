from dataclasses import dataclass

import numpy as np
import gc
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DiffusionPipeline
from contextlib import contextmanager
from jaxtyping import Float
from typing import Literal
from PIL import Image
import matplotlib.pyplot as plt
# from utils.imageutil import clip_image_at_percentiles
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.models.embeddings import TimestepEmbedding

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

class ToWeightsDType(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        return self.module(x).to(self.dtype)

@dataclass
class GuidanceConfig:
    sd_pretrained_model_or_path: str = "runwayml/stable-diffusion-v2-1-base"
    sd_pretrained_model_or_path_lora: str = "stabilityai/stable-diffusion-2-1"

    num_inference_steps: int = 500
    min_step_ratio: float = 0.02
    max_step_ratio: float = 0.98

    src_prompt: str = "a photo of a man"
    tgt_prompt: str = "a photo of a man raising his arms"

    guidance_scale: float = 100
    guidance_scale_lora: float = 1.0
    sdedit_guidance_scale: float = 15
    device: torch.device = torch.device("cuda")
    lora_n_timestamp_samples: int = 1

    sync_noise_and_t: bool = True
    lora_cfg_training: bool = True


class Guidance(object):
    def __init__(self, config: GuidanceConfig, use_lora: bool = False):
        self.config = config
        self.device = torch.device(config.device)

        self.pipe = DiffusionPipeline.from_pretrained(config.sd_pretrained_model_or_path).to(self.device)

        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler.set_timesteps(config.num_inference_steps)
        self.pipe.scheduler = self.scheduler

        self.unet = self.pipe.unet
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae

        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        ## construct text features beforehand.
        self.src_prompt = self.config.src_prompt
        self.tgt_prompt = self.config.tgt_prompt

        self.update_text_features(src_prompt=self.src_prompt, tgt_prompt=self.tgt_prompt)
        self.null_text_feature = self.encode_text("")

        if use_lora:
            self.pipe_lora = DiffusionPipeline.from_pretrained(config.sd_pretrained_model_or_path_lora).to(self.device)
            self.single_model = False
            del self.pipe_lora.vae
            del self.pipe_lora.text_encoder
            cleanup()
            self.vae_lora = self.pipe_lora.vae = self.pipe.vae
            self.unet_lora = self.pipe_lora.unet
            for p in self.unet_lora.parameters():
                p.requires_grad_(False)
            # FIXME: hard-coded dims
            self.camera_embedding = TimestepEmbedding(16, 1280).to(self.device)
            self.unet_lora.class_embedding = self.camera_embedding
            self.scheduler_lora = DDIMScheduler.from_config(self.pipe_lora.scheduler.config)
            self.scheduler_lora.set_timesteps(config.num_inference_steps)
            self.pipe_lora.scheduler = self.scheduler_lora

            # set up LoRA layers
            lora_attn_procs = {}
            for name in self.unet_lora.attn_processors.keys():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else self.unet_lora.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = self.unet_lora.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet_lora.config.block_out_channels))[
                        block_id
                    ]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet_lora.config.block_out_channels[block_id]

                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
                )

            self.unet_lora.set_attn_processor(lora_attn_procs)

            self.lora_layers = AttnProcsLayers(self.unet_lora.attn_processors).to(
                self.device
            )
            self.lora_layers._load_state_dict_pre_hooks.clear()
            self.lora_layers._state_dict_hooks.clear()

    def compute_posterior_mean(self, xt, noise_pred, t, t_prev):
        """
        Computes an estimated posterior mean \mu_\phi(x_t, y; \epsilon_\phi).
        """
        device = self.device
        beta_t = self.scheduler.betas[t].to(device)[:, None, None, None]
        alpha_t = self.scheduler.alphas[t].to(device)[:, None, None, None]
        alpha_bar_t = self.scheduler.alphas_cumprod[t].to(device)[:, None, None, None]
        alpha_bar_t_prev = self.scheduler.alphas_cumprod[t_prev].to(device)[:, None, None, None]

        pred_x0 = (xt - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        c0 = torch.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)
        c1 = torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)

        mean_func = c0 * pred_x0 + c1 * xt
        return mean_func

    def encode_image(self, img_tensor: Float[torch.Tensor, "B C H W"]):
        x = img_tensor
        x = 2 * x - 1
        x = x.float()
        return self.vae.encode(x).latent_dist.sample() * 0.18215

    def encode_text(self, prompt):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_encoding = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_encoding

    def decode_latent(self, latent):
        x = self.vae.decode(latent / 0.18215).sample
        x = (x / 2 + 0.5).clamp(0, 1)
        return x

    def update_text_features(self, src_prompt=None, tgt_prompt=None):
        if getattr(self, "src_text_feature", None) is None:
            assert src_prompt is not None
            self.src_prompt = src_prompt
            self.src_text_feature = self.encode_text(src_prompt)
        else:
            if src_prompt is not None and src_prompt != self.src_prompt:
                self.src_prompt = src_prompt
                self.src_text_feature = self.encode_text(src_prompt)

        if getattr(self, "tgt_text_feature", None) is None:
            assert tgt_prompt is not None
            self.tgt_prompt = tgt_prompt
            self.tgt_text_feature = self.encode_text(tgt_prompt)
        else:
            if tgt_prompt is not None and tgt_prompt != self.tgt_prompt:
                self.tgt_prompt = tgt_prompt
                self.tgt_text_feature = self.encode_text(tgt_prompt)

    def pds_timestep_sampling(self, batch_size):
        self.scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = reversed(self.scheduler.timesteps)

        min_step = 1 if self.config.min_step_ratio <= 0 else int(len(timesteps) * self.config.min_step_ratio)
        max_step = (
            len(timesteps) if self.config.max_step_ratio >= 1 else int(len(timesteps) * self.config.max_step_ratio)
        )
        max_step = max(max_step, min_step + 1)
        idx = torch.randint(
            min_step,
            max_step,
            [batch_size],
            dtype=torch.long,
            device="cpu",
        )
        t = timesteps[idx].cpu()
        t_prev = timesteps[idx - 1].cpu()

        return t, t_prev

    def __call__(
        self,
        tgt_x0,
        src_x0,
        tgt_prompt=None,
        src_prompt=None,
        reduction="mean",
        return_dict=False,
    ):
        device = self.device
        scheduler = self.scheduler

        # process text.
        self.update_text_features(src_prompt=src_prompt, tgt_prompt=tgt_prompt)
        tgt_text_embedding, src_text_embedding = (
            self.tgt_text_feature,
            self.src_text_feature,
        )
        uncond_embedding = self.null_text_feature

        batch_size = tgt_x0.shape[0]
        t, t_prev = self.pds_timestep_sampling(batch_size)
        beta_t = scheduler.betas[t].to(device)
        alpha_bar_t = scheduler.alphas_cumprod[t].to(device)
        alpha_bar_t_prev = scheduler.alphas_cumprod[t_prev].to(device)
        sigma_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t) ** (0.5)

        noise = torch.randn_like(tgt_x0)
        noise_t_prev = torch.randn_like(tgt_x0)

        zts = dict()
        for latent, cond_text_embedding, name in zip(
            [tgt_x0, src_x0], [tgt_text_embedding, src_text_embedding], ["tgt", "src"]
        ):
            latents_noisy = scheduler.add_noise(latent, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            text_embeddings = torch.cat([cond_text_embedding, uncond_embedding], dim=0)
            noise_pred = self.unet.forward(
                latent_model_input,
                torch.cat([t] * 2).to(device),
                encoder_hidden_states=text_embeddings,
            ).sample
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_uncond)

            x_t_prev = scheduler.add_noise(latent, noise_t_prev, t_prev)
            mu = self.compute_posterior_mean(latents_noisy, noise_pred, t, t_prev)
            zt = (x_t_prev - mu) / sigma_t
            zts[name] = zt

        grad = zts["tgt"] - zts["src"]
        grad = torch.nan_to_num(grad)
        target = (tgt_x0 - grad).detach()
        loss = 0.5 * F.mse_loss(tgt_x0, target, reduction=reduction) / batch_size
        if return_dict:
            dic = {"loss": loss, "grad": grad, "t": t}
            return dic
        else:
            return loss



guidance = Guidance(GuidanceConfig(
    sd_pretrained_model_or_path='stabilityai/stable-diffusion-2-1-base'
), use_lora=False)

import ipdb; ipdb.set_trace()


latents_T = torch.randn((1, 4, 64, 64), device=guidance.unet.device)
image = guidance.pipe(prompt="A cat", negative_prompt="", guidance_scale=1, num_inference_steps=50, latents=latents_T)
image2 = guidance.pipe(prompt="A dog", negative_prompt="", guidance_scale=1, num_inference_steps=50, latents=latents_T)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image.images[0])
ax[1].imshow(image2.images[0])
plt.savefig('results/figure2/cat_dog_sample_cfg7.5.png')
plt.savefig('results/figure2/cat_dog_sample_cfg1.png')

image = guidance.pipe(prompt="A cat", negative_prompt="", guidance_scale=1, num_inference_steps=999, latents=latents_T, output_type='latent')
image2 = guidance.pipe(prompt="A dog", negative_prompt="", guidance_scale=1, num_inference_steps=999, latents=latents_T, output_type='latent')

gt_direction = (image[0] - image2[0]).reshape(-1)
gt_direction_normalized_999 = gt_direction / torch.norm(gt_direction)


guidance.update_text_features(src_prompt='A cat', tgt_prompt='A dog')
tgt_text_embedding, src_text_embedding = (guidance.tgt_text_feature, guidance.src_text_feature)
latent_model_input = torch.cat([latents_T] * 2, dim=0)
text_embeddings = torch.cat([tgt_text_embedding, src_text_embedding], dim=0)
noise_pred = guidance.unet.forward(latent_model_input, torch.cat([torch.Tensor([999]).long()] * 2).to('cuda'), encoder_hidden_states=text_embeddings).sample
noise_pred_tgt, noise_pred_src = noise_pred.chunk(2)

approx_direction  = (noise_pred_tgt - noise_pred_src).reshape(-1)
approx_direction_normalized = approx_direction / torch.norm(approx_direction)

cosine_similarity = torch.dot(gt_direction_normalized_999, approx_direction_normalized)


guidance.update_text_features(src_prompt='', tgt_prompt='A dog')
tgt_text_embedding, src_text_embedding = (guidance.tgt_text_feature, guidance.src_text_feature)
latent_model_input = torch.cat([latents_T] * 2, dim=0)
text_embeddings = torch.cat([tgt_text_embedding, src_text_embedding], dim=0)
noise_pred = guidance.unet.forward(latent_model_input, torch.cat([torch.Tensor([999]).long()] * 2).to('cuda'), encoder_hidden_states=text_embeddings).sample
noise_pred_tgt, noise_pred_src = noise_pred.chunk(2)
uncond_direction  = (noise_pred_tgt - noise_pred_src).reshape(-1)
uncond_direction_normalized = uncond_direction / torch.norm(uncond_direction)

cosine_similarity2 = torch.dot(gt_direction_normalized_999, uncond_direction_normalized)


noise = torch.randn_like(latents_T)
for i in range(0, 999, 100):
    t = i + 1
    latents_noisy = guidance.scheduler.add_noise(image[0], noise, torch.Tensor([t]).long())
    # uncond
    guidance.update_text_features(src_prompt='A cat', tgt_prompt='A dog')
    tgt_text_embedding, src_text_embedding = (guidance.tgt_text_feature, guidance.src_text_feature)
    latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
    text_embeddings = torch.cat([tgt_text_embedding, src_text_embedding], dim=0)
    noise_pred = guidance.unet.forward(latent_model_input, torch.cat([torch.Tensor([t]).long()] * 2).to('cuda'), encoder_hidden_states=text_embeddings).sample
    noise_pred_tgt, noise_pred_src = noise_pred.chunk(2)
    approx_direction  = (noise_pred_tgt - noise_pred_src).reshape(-1)
    approx_direction_normalized = approx_direction / torch.norm(approx_direction)
    cosine_similarity = torch.dot(gt_direction_normalized, approx_direction_normalized)
    # uncond
    guidance.update_text_features(src_prompt='', tgt_prompt='A dog')
    tgt_text_embedding, src_text_embedding = (guidance.tgt_text_feature, guidance.src_text_feature)
    latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
    text_embeddings = torch.cat([tgt_text_embedding, src_text_embedding], dim=0)
    noise_pred = guidance.unet.forward(latent_model_input, torch.cat([torch.Tensor([t]).long()] * 2).to('cuda'), encoder_hidden_states=text_embeddings).sample
    noise_pred_tgt, noise_pred_src = noise_pred.chunk(2)
    uncond_direction  = (noise_pred_tgt - noise_pred_src).reshape(-1)
    uncond_direction_normalized = uncond_direction / torch.norm(uncond_direction)
    cosine_similarity2 = torch.dot(gt_direction_normalized, uncond_direction_normalized)
    print(f"t={t}, cosine_similarity={cosine_similarity}, cosine_similarity2={cosine_similarity2}")