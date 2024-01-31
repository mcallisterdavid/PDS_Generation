from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DiffusionPipeline
from jaxtyping import Float
from typing import Literal
from PIL import Image
import matplotlib.pyplot as plt
from utils.imageutil import clip_image_at_percentiles

@dataclass
class PDSConfig:
    sd_pretrained_model_or_path: str = "runwayml/stable-diffusion-v1-5"

    num_inference_steps: int = 500
    min_step_ratio: float = 0.02
    max_step_ratio: float = 0.98

    src_prompt: str = "a photo of a man"
    tgt_prompt: str = "a photo of a man raising his arms"

    guidance_scale: float = 100
    sdedit_guidance_scale: float = 15
    device: torch.device = torch.device("cuda")


class PDS(object):
    def __init__(self, config: PDSConfig):
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
        src_cfg_scale=100,
        tgt_cfg_scale=100,
        src_method: Literal["sdedit", "step"] = "sdedit",
        reduction="mean",
        return_dict=False,
    ):
        device = self.device
        scheduler = self.scheduler

        # process text.
        self.update_text_features(tgt_prompt=prompt + ', detailed, high resolution, high quality, sharp', src_prompt=prompt + ', pixelated, foggy, hazy, blurry, noisy, malformed')
        tgt_text_embedding = self.tgt_text_feature
        src_text_embedding = self.src_text_feature
        uncond_embedding = self.null_text_feature

        tgt_x0 = im

        if src_method == "sdedit":
            src_x0 = self.run_sdedit(
                x0=im,
                tgt_prompt=prompt,
                num_inference_steps=num_solve_steps,
                skip=int(skip_percentage * num_solve_steps),
            )
        elif src_method == "step":
            lower_bound =  999 - int(skip_percentage * 1000)
            upper_bound = 1000 - int(skip_percentage * 950)
            src_x0 = self.run_single_step(
                x0=im,
                t = torch.randint(lower_bound, upper_bound, size=(1,)).item(),
                tgt_prompt=prompt,
            )

        batch_size = im.shape[0]
        t, t_prev = self.pds_timestep_sampling(batch_size)
        beta_t = scheduler.betas[t].to(device)
        alpha_bar_t = scheduler.alphas_cumprod[t].to(device)
        alpha_bar_t_prev = scheduler.alphas_cumprod[t_prev].to(device)
        sigma_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t) ** (0.5)

        noise = torch.randn_like(tgt_x0)
        noise_t_prev = torch.randn_like(tgt_x0)

        zts = dict()
        for latent, cond_text_embedding, name, cfg_scale in zip(
            [tgt_x0, src_x0], [tgt_text_embedding, src_text_embedding], ["tgt", "src"], [tgt_cfg_scale, src_cfg_scale]
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
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

            x_t_prev = scheduler.add_noise(latent, noise_t_prev, t_prev)
            mu = self.compute_posterior_mean(latents_noisy, noise_pred, t, t_prev)
            zt = (x_t_prev - mu) / sigma_t
            zts[name] = zt

        grad = zts["tgt"] - zts["src"]
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
        noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_uncond)

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

    def run_sdedit(self, x0, tgt_prompt=None, num_inference_steps=20, skip=7, eta=0):
        scheduler = self.scheduler
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps
        reversed_timesteps = reversed(scheduler.timesteps)

        S = num_inference_steps - skip
        t = reversed_timesteps[S - 1]
        noise = torch.randn_like(x0)

        xt = scheduler.add_noise(x0, noise, t)

        self.update_text_features(None, tgt_prompt=tgt_prompt)
        tgt_text_embedding = self.tgt_text_feature
        null_text_embedding = self.null_text_feature
        text_embeddings = torch.cat([tgt_text_embedding, null_text_embedding], dim=0)

        op = timesteps[-S:]

        for t in op:
            xt_input = torch.cat([xt] * 2)
            noise_pred = self.unet.forward(
                xt_input,
                torch.cat([t[None]] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            ).sample
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            # noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_uncond)
            noise_pred = noise_pred_uncond + self.config.sdedit_guidance_scale * (noise_pred_text - noise_pred_uncond)
            xt = self.reverse_step(noise_pred, t, xt, eta=eta, scheduler=scheduler)

        return xt

    def run_single_step(self, x0, t: int, tgt_prompt=None, percentile_clip=False, eta=0):
        scheduler = self.scheduler

        t = torch.tensor(t, device=x0.device)
        noise = torch.randn_like(x0)

        xt = scheduler.add_noise(x0, noise, t)

        self.update_text_features(None, tgt_prompt=tgt_prompt)
        tgt_text_embedding = self.tgt_text_feature
        null_text_embedding = self.null_text_feature
        text_embeddings = torch.cat([tgt_text_embedding, null_text_embedding], dim=0)

        xt_input = torch.cat([xt] * 2)
        noise_pred = self.unet.forward(
            xt_input,
            torch.cat([t[None]] * 2).to(self.device),
            encoder_hidden_states=text_embeddings,
        ).sample
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.config.sdedit_guidance_scale * (noise_pred_text - noise_pred_uncond)

        xhat_pred = self.scheduler.step(noise_pred, t, xt, eta=eta).pred_original_sample

        if percentile_clip:
            xhat_pred = clip_image_at_percentiles(xhat_pred, 0.05, 0.95)

        return xhat_pred

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
