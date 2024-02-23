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
from utils.imageutil import clip_image_at_percentiles
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
        
    def pred_for_x_hat(
        self,
        pred_original_sample: torch.FloatTensor,
        t: int,
        sample: torch.FloatTensor,
        scheduler,
    ):
        device = sample.device
        if scheduler.config.prediction_type == "v_prediction":
            alpha_prod_t = scheduler.alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            v_pred = (sample * alpha_prod_t ** (0.5) - pred_original_sample) / (beta_prod_t ** (0.5))
            pred = v_pred
        elif scheduler.config.prediction_type == "epsilon":
            alpha_prod_t = scheduler.alphas_cumprod[t].to(device)
            alpha_prod_t = alpha_prod_t[:, None, None, None]
            beta_prod_t = 1 - alpha_prod_t
            v_pred = sample * alpha_prod_t - pred_original_sample
            v_pred = (sample * alpha_prod_t ** (0.5) - pred_original_sample) / (beta_prod_t ** (0.5))
            pred_epsilon = (alpha_prod_t**0.5) * v_pred + (beta_prod_t**0.5) * sample
            pred = pred_epsilon
        else:
            raise ValueError()

        return pred

    def pds_gen(
        self,
        im,
        prompt=None,
        reduction="mean",
        return_dict=False,
    ):
        device = self.device
        scheduler = self.scheduler

        # process text.
        self.update_text_features(tgt_prompt=prompt)
        tgt_text_embedding = self.tgt_text_feature
        uncond_embedding = self.null_text_feature

        batch_size = im.shape[0]
        t, t_prev = self.pds_timestep_sampling(batch_size)
        beta_t = scheduler.betas[t].to(device)
        alpha_bar_t = scheduler.alphas_cumprod[t].to(device)
        alpha_bar_t_prev = scheduler.alphas_cumprod[t_prev].to(device)
        sigma_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t) ** (0.5)

        noise = torch.randn_like(im)
        noise_t_prev = torch.randn_like(im)

        latents_noisy = scheduler.add_noise(im, noise, t)
        latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
        text_embeddings = torch.cat([tgt_text_embedding, uncond_embedding] * batch_size, dim=0)

        noise_pred = self.unet.forward(
            latent_model_input,
            torch.cat([t] * 2).to(device),
            encoder_hidden_states=text_embeddings,
        ).sample
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_uncond)

        x_t_prev = scheduler.add_noise(im, noise_t_prev, t_prev)

        noise_pred_src = self.pred_for_x_hat(im, t, latents_noisy, scheduler)

        mu_src = self.compute_posterior_mean(latents_noisy, noise_pred_src, t, t_prev)
        mu_tgt = self.compute_posterior_mean(latents_noisy, noise_pred, t, t_prev)

        zt_src = (x_t_prev - mu_src) / sigma_t[:, None, None, None]
        zt_tgt = (x_t_prev - mu_tgt) / sigma_t[:, None, None, None]

        grad = zt_tgt - zt_src
        grad = torch.nan_to_num(grad)
        target = (im - grad).detach()
        loss = 0.5 * F.mse_loss(im, target, reduction=reduction) / batch_size
        if return_dict:
            dic = {"loss": loss, "grad": grad, "t": t}
            return dic
        else:
            return loss


    def pds_gen_sdedit_src(
        self,
        im,
        prompt=None,
        skip_percentage=0.5,
        num_solve_steps=20,
        tgt_prompt='detailed, high resolution, high quality, sharp',
        src_prompt='pixelated, foggy, hazy, blurry, noisy, malformed',
        src_cfg_scale=100,
        tgt_cfg_scale=100,
        src_method: Literal["sdedit", "step"] = "sdedit",
        grad_method="z",
        reduction="mean",
        noise = None,
        return_dict=False,
    ):
        device = self.device
        scheduler = self.scheduler

        # process text.
        self.update_text_features(tgt_prompt=prompt + ', ' + tgt_prompt, src_prompt=prompt + ', ' + src_prompt)
        # self.update_text_features(tgt_prompt=prompt, src_prompt=prompt)
        tgt_text_embedding = self.tgt_text_feature
        src_text_embedding = self.src_text_feature
        uncond_embedding = self.null_text_feature

        tgt_x0 = im

        if src_method == "sdedit":
            lower_bound =  20
            upper_bound = 980
            t_proj = torch.randint(lower_bound, upper_bound, size=(1,))
            src_x0 = im
            src_x0, noise_proj, t_proj = self.run_sdedit(
                x0=src_x0,
                tgt_prompt=prompt,
                noise=noise,
                t=t_proj,
                num_inference_steps=100,
            )
        elif src_method == "step":
            # lower_bound =  999 - int(skip_percentage * 1000)
            # upper_bound = 1000 - int(skip_percentage * 950)
            lower_bound =  20
            upper_bound = 980
            t_proj = torch.randint(lower_bound, upper_bound, size=(1,))
            src_x0 = im
            src_x0, noise_proj, t_proj = self.run_single_step(
                x0=src_x0,
                t=t_proj,
                noise=noise,
                tgt_prompt=prompt,
            )
        elif src_method == "ddpm":
            t =  int(skip_percentage * 1000)
            src_x0, noise_proj, t_proj = self.run_single_step(
                x0=im,
                t = torch.ones((1,), dtype=torch.long)*t,
                tgt_prompt=prompt,
            )
        elif src_method == "copy":
            src_x0 = tgt_x0.clone().detach()

        if self.config.sync_noise_and_t:
            noise = noise_proj
            t = t_proj
            t_prev = t - 2
        else:
            noise = torch.randn_like(tgt_x0)
            t, t_prev = self.pds_timestep_sampling(batch_size)

        batch_size = im.shape[0]
        beta_t = scheduler.betas[t].to(device)
        alpha_bar_t = scheduler.alphas_cumprod[t].to(device)
        alpha_bar_t_prev = scheduler.alphas_cumprod[t_prev].to(device)
        sigma_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t) ** (0.5)

        noise_t_prev = torch.randn_like(tgt_x0)

        zts = dict()
        x0s = dict()
        predicted_eps = dict()
        for latent, cond_text_embedding, name, cfg_scale in zip(
            [tgt_x0, src_x0], [tgt_text_embedding, src_text_embedding], ["tgt", "src"], [tgt_cfg_scale, src_cfg_scale]
        ):
            if grad_method.startswith("iid"):
                src_x0_noisy = scheduler.add_noise(src_x0, noise, t)
                latents_noisy = scheduler.add_noise(latent, noise, t)
                src_x0_model_input = torch.cat([src_x0_noisy] * 2, dim=0)
                text_embeddings = torch.cat([cond_text_embedding, uncond_embedding], dim=0)
                noise_pred = self.unet.forward(
                    src_x0_model_input,
                    torch.cat([t] * 2).to(device),
                    encoder_hidden_states=text_embeddings,
                ).sample
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

            elif grad_method.startswith("ood"):
                tgt_x0_noisy = scheduler.add_noise(tgt_x0, noise, t)
                latents_noisy = scheduler.add_noise(latent, noise, t)
                tgt_x0_model_input = torch.cat([tgt_x0_noisy] * 2, dim=0)
                text_embeddings = torch.cat([cond_text_embedding, uncond_embedding], dim=0)
                noise_pred = self.unet.forward(
                    tgt_x0_model_input,
                    torch.cat([t] * 2).to(device),
                    encoder_hidden_states=text_embeddings,
                ).sample
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

            else:
                # this is to compute epsilon both on corresponding x0
                latents_noisy = scheduler.add_noise(latent, noise, t)
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                text_embeddings = torch.cat([cond_text_embedding, uncond_embedding], dim=0)
                noise_pred = self.unet.forward(
                    latent_model_input,
                    torch.cat([t] * 2).to(device),
                    encoder_hidden_states=text_embeddings,
                ).sample
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

            if grad_method.endswith("z"):
                x_t_prev = scheduler.add_noise(latent, noise_t_prev, t_prev)
                mu = self.compute_posterior_mean(latents_noisy, noise_pred, t, t_prev)
                zt = (x_t_prev - mu) / sigma_t
                zts[name] = zt
            elif grad_method.endswith("eps"):
                zts[name] = noise_pred
            elif grad_method.endswith("x"):
                zts[name] = latent
            elif grad_method.endswith("xps"):
                zts[name] = latent + noise_pred
            elif grad_method == "condonly":
                zts[name] = cfg_scale * noise_pred_text
            elif grad_method == "uncondonly":
                zts[name] = (1-cfg_scale) * noise_pred_text
            else:
                raise NotImplementedError(f"grad_method {grad_method} not implemented")
            x0s[name] = latent
            predicted_eps[name] = noise_pred
        #     print(name, t, 'latent norm: %.2f' % torch.norm(latent).item(), 'noise_pred norm: %.2f' % torch.norm(noise_pred).item(), 'ratio: %.2f' % (torch.norm(noise_pred) / torch.norm(latent)).item())

            
        # print(t, 'latent grad norm: %.2f' % torch.norm(x0s["tgt"] - x0s["src"]).item())
        # print(t, 'eps grad norm: %.2f' % torch.norm(predicted_eps["tgt"] - predicted_eps["src"]).item())
        grad = zts["tgt"] - zts["src"]

        if 'sds' in grad_method:
            self.update_text_features(tgt_prompt=prompt)
            tgt_text_embedding = self.tgt_text_feature
            uncond_embedding = self.null_text_feature
            noise_sds = torch.randn_like(im)

            if 'sds2' in grad_method:
                latents_noisy = scheduler.add_noise(im, noise_sds, t)
            else:
                latents_noisy = scheduler.add_noise(src_x0, noise_sds, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            text_embeddings = torch.cat([tgt_text_embedding, uncond_embedding], dim=0)
            noise_pred = self.unet.forward(
                latent_model_input,
                torch.cat([t] * 2).to(device),
                encoder_hidden_states=text_embeddings,
            ).sample
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            w = 1 - scheduler.alphas_cumprod[t].to(device)
            grad += w * (noise_pred - noise_sds)

        grad = torch.nan_to_num(grad)
        target = (tgt_x0 - grad).detach()
        loss = 0.5 * F.mse_loss(tgt_x0, target, reduction=reduction) / batch_size
        if return_dict:
            dic = {"loss": loss, "grad": grad, "t": t, "src_x0": src_x0}
            return dic
        else:
            return loss

    def sds_loss(
        self,
        im,
        prompt=None,
        reduction="mean",
        cfg_scale=100,
        noise=None,
        return_dict=False,
    ):
        device = self.device
        scheduler = self.scheduler

        # process text.
        self.update_text_features(tgt_prompt=prompt)
        tgt_text_embedding = self.tgt_text_feature
        uncond_embedding = self.null_text_feature

        batch_size = im.shape[0]
        t, _ = self.pds_timestep_sampling(batch_size)

        if noise is None:
            noise = torch.randn_like(im)

        latents_noisy = scheduler.add_noise(im, noise, t)
        latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
        text_embeddings = torch.cat([tgt_text_embedding, uncond_embedding], dim=0)
        noise_pred = self.unet.forward(
            latent_model_input,
            torch.cat([t] * 2).to(device),
            encoder_hidden_states=text_embeddings,
        ).sample
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

        w = 1 - scheduler.alphas_cumprod[t].to(device)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        target = (im - grad).detach()
        loss = 0.5 * F.mse_loss(im, target, reduction=reduction) / batch_size
        if return_dict:
            dic = {"loss": loss, "grad": grad, "t": t}
            return dic
        else:
            return loss

    def nfsd_loss(
        self,
        im,
        prompt=None,
        reduction="mean",
        cfg_scale=100,
        return_dict=False,
    ):
        device = self.device
        scheduler = self.scheduler


        batch_size = im.shape[0]
        t, _ = self.pds_timestep_sampling(batch_size)

        noise = torch.randn_like(im)

        latents_noisy = scheduler.add_noise(im, noise, t)
        latent_model_input = torch.cat([latents_noisy] * 2, dim=0)

        # process text.
        self.update_text_features(tgt_prompt=prompt)
        tgt_text_embedding = self.tgt_text_feature
        uncond_embedding = self.null_text_feature
        with torch.no_grad():
            text_embeddings = torch.cat([tgt_text_embedding, uncond_embedding], dim=0)
            noise_pred = self.unet.forward(
                latent_model_input,
                torch.cat([t] * 2).to(device),
                encoder_hidden_states=text_embeddings,
            ).sample
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            delta_C = cfg_scale * (noise_pred_text - noise_pred_uncond)

        self.update_text_features(tgt_prompt='unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, dark, low-resolution, gloomy')
        tgt_text_embedding = self.tgt_text_feature
        uncond_embedding = self.null_text_feature
        with torch.no_grad():
            text_embeddings = torch.cat([tgt_text_embedding, uncond_embedding], dim=0)
            noise_pred = self.unet.forward(
                latent_model_input,
                torch.cat([t] * 2).to(device),
                encoder_hidden_states=text_embeddings,
            ).sample
            noise_pred_text_neg, _ = noise_pred.chunk(2)

        delta_D = noise_pred_uncond if t < 200 else (noise_pred_uncond - noise_pred_text_neg)

        w = 1 - scheduler.alphas_cumprod[t].to(device)
        grad = w * (delta_C + delta_D)
        grad = torch.nan_to_num(grad)
        target = (im - grad).detach()
        loss = 0.5 * F.mse_loss(im, target, reduction=reduction) / batch_size
        if return_dict:
            dic = {"loss": loss, "grad": grad, "t": t}
            return dic
        else:
            return loss

    def run_sdedit(self, x0, t=None, noise=None, tgt_prompt=None, num_inference_steps=1000, skip=7, eta=0):
        scheduler = self.scheduler
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps
        reversed_timesteps = reversed(scheduler.timesteps)

        if t is None:
            S = num_inference_steps - skip
            t = reversed_timesteps[S - 1]
        else:
            S = min(range(len(reversed_timesteps)), key=lambda i: abs(reversed_timesteps[i] - t)) + 1

        if noise is None:
            noise = torch.randn_like(x0)

        xt = scheduler.add_noise(x0, noise, t)
        self.update_text_features(None, tgt_prompt=tgt_prompt)
        tgt_text_embedding = self.tgt_text_feature
        null_text_embedding = self.null_text_feature
        text_embeddings = torch.cat([tgt_text_embedding, null_text_embedding], dim=0)

        op = timesteps[-S:]

        for step in op:
            xt_input = torch.cat([xt] * 2)
            noise_pred = self.unet.forward(
                xt_input,
                torch.cat([step[None]] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            ).sample
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            # noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_uncond)
            noise_pred = noise_pred_uncond + self.config.sdedit_guidance_scale * (noise_pred_text - noise_pred_uncond)
            xt = self.reverse_step(noise_pred, step, xt, eta=eta, scheduler=scheduler)

        return xt, noise, t

    def run_single_step(self, x0, t: int, noise=None, tgt_prompt=None, percentile_clip=False, eta=0):
        scheduler = self.scheduler

        if noise is None:
            noise = torch.randn_like(x0)

        xt = scheduler.add_noise(x0, noise, t)

        self.update_text_features(None, tgt_prompt=tgt_prompt)
        tgt_text_embedding = self.tgt_text_feature
        null_text_embedding = self.null_text_feature
        text_embeddings = torch.cat([tgt_text_embedding, null_text_embedding], dim=0)

        xt_input = torch.cat([xt] * 2)
        noise_pred = self.unet.forward(
            xt_input,
            torch.cat([t] * 2).to(self.device),
            encoder_hidden_states=text_embeddings,
        ).sample
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.config.sdedit_guidance_scale * (noise_pred_text - noise_pred_uncond)

        xhat_pred = self.scheduler.step(noise_pred, t.item(), xt, eta=eta).pred_original_sample

        if percentile_clip:
            xhat_pred = clip_image_at_percentiles(xhat_pred, 0.05, 0.95)

        return xhat_pred, noise, t

    def reverse_step(self, model_output, timestep, sample, eta=0, variance_noise=None, scheduler=None):

        if scheduler is None:
            scheduler = self.scheduler

        prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        alpha_prod_t = scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        variance = self.get_variance(timestep, scheduler)
        model_output_direction = model_output
        pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (0.5) * model_output_direction
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        if eta > 0:
            if variance_noise is None:
                variance_noise = torch.randn_like(model_output)
            sigma_z = eta * variance ** (0.5) * variance_noise
            prev_sample = prev_sample + sigma_z
        return prev_sample

    def get_variance(self, timestep, scheduler=None):

        if scheduler is None:
            scheduler = self.scheduler

        prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    @contextmanager
    def disable_unet_class_embedding(self, unet):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding


    def vsd_loss(
        self,
        im,
        prompt=None,
        reduction="mean",
        cfg_scale=100,
        return_dict=False,
    ):
        device = self.device
        scheduler = self.scheduler

        # process text.
        self.update_text_features(tgt_prompt=prompt)
        tgt_text_embedding = self.tgt_text_feature
        uncond_embedding = self.null_text_feature

        batch_size = im.shape[0]
        camera_condition = torch.zeros([batch_size, 4, 4], device=device)
        
        with torch.no_grad():
            # random timestamp
            t = torch.randint(
                20,
                980 + 1,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )
            
            noise = torch.randn_like(im)

            latents_noisy = scheduler.add_noise(im, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            text_embeddings = torch.cat([tgt_text_embedding, uncond_embedding], dim=0)
            with self.disable_unet_class_embedding(self.unet) as unet:
                cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
                noise_pred_pretrain = unet.forward(
                    latent_model_input,
                    torch.cat([t] * 2).to(device),
                    encoder_hidden_states=text_embeddings,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

            # use view-independent text embeddings in LoRA
            noise_pred_est = self.unet_lora.forward(
                latent_model_input,
                torch.cat([t] * 2).to(device),
                encoder_hidden_states=torch.cat([tgt_text_embedding] * 2),
                class_labels=torch.cat(
                    [
                        camera_condition.view(batch_size, -1),
                        camera_condition.view(batch_size, -1),
                    ],
                    dim=0,
                ),
                cross_attention_kwargs={"scale": 1.0},
            ).sample

        (
            noise_pred_pretrain_text,
            noise_pred_pretrain_uncond,
        ) = noise_pred_pretrain.sample.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_pretrain = noise_pred_pretrain_uncond + cfg_scale * (
            noise_pred_pretrain_text - noise_pred_pretrain_uncond
        )
        assert self.scheduler.config.prediction_type == "epsilon"
        if self.scheduler_lora.config.prediction_type == "v_prediction":
            alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(
                device=latents_noisy.device, dtype=latents_noisy.dtype
            )
            alpha_t = alphas_cumprod[t] ** 0.5
            sigma_t = (1 - alphas_cumprod[t]) ** 0.5

            noise_pred_est = latent_model_input * torch.cat([sigma_t] * 2, dim=0).view(
                -1, 1, 1, 1
            ) + noise_pred_est * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)

        (
            noise_pred_est_camera,
            noise_pred_est_uncond,
        ) = noise_pred_est.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_est = noise_pred_est_uncond + self.config.guidance_scale_lora * (
            noise_pred_est_camera - noise_pred_est_uncond
        )

        w = (1 - scheduler.alphas_cumprod[t.cpu()]).view(-1, 1, 1, 1).to(device)
        grad = w * (noise_pred_pretrain - noise_pred_est)

        grad = torch.nan_to_num(grad)
        target = (im - grad).detach()
        loss = 0.5 * F.mse_loss(im, target, reduction=reduction) / batch_size
        loss_lora = self.train_lora(im, text_embeddings, camera_condition)
        if return_dict:
            dic = {"loss": loss, "lora_loss": loss_lora, "grad": grad, "t": t}
            return dic
        else:
            return loss

    def train_lora(
        self,
        latents: Float[torch.Tensor, "B 4 64 64"],
        text_embeddings: Float[torch.Tensor, "BB 77 768"],
        camera_condition: Float[torch.Tensor, "B 4 4"],
    ):
        scheduler = self.scheduler_lora

        B = latents.shape[0]
        latents = latents.detach().repeat(self.config.lora_n_timestamp_samples, 1, 1, 1)

        t = torch.randint(
            int(scheduler.num_train_timesteps * 0.0),
            int(scheduler.num_train_timesteps * 1.0),
            [B * self.config.lora_n_timestamp_samples],
            dtype=torch.long,
            device=self.device,
        )

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler_lora.add_noise(latents, noise, t)
        if self.scheduler_lora.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler_lora.config.prediction_type == "v_prediction":
            target = self.scheduler_lora.get_velocity(latents, noise, t)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler_lora.config.prediction_type}"
            )
        # use view-independent text embeddings in LoRA
        text_embeddings_cond, _ = text_embeddings.chunk(2)
        if self.config.lora_cfg_training and np.random.random() < 0.1:
            camera_condition = torch.zeros_like(camera_condition)
        noise_pred = self.unet_lora.forward(
            noisy_latents,
            t,
            encoder_hidden_states=text_embeddings_cond.repeat(
                self.config.lora_n_timestamp_samples, 1, 1
            ),
            class_labels=camera_condition.view(B, -1).repeat(
                self.config.lora_n_timestamp_samples, 1
            ),
            cross_attention_kwargs={"scale": 1.0},
        ).sample
        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")


def tensor_to_pil(img):
    if img.ndim == 4:
        img = img[0]
    img = img.cpu().permute(1, 2, 0).detach().numpy()
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    return img


def pil_to_tensor(img, device="cpu"):
    device = torch.device(device)
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img[None].transpose(0, 3, 1, 2))
    img = img.to(device)
    return img
