from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DiffusionPipeline
from jaxtyping import Float
from typing import Literal, Optional, Tuple, Union
from PIL import Image
import matplotlib.pyplot as plt

@dataclass
class PDSConfig:
    sd_pretrained_model_or_path: str = "runwayml/stable-diffusion-v1-5"
    texture_inversion_embedding: str = "./pds/assets/learned_embeds-steps-1500.safetensors"

    # num_inference_steps: int = 500
    num_inference_steps: int = 50
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
        self.pipe.load_textual_inversion(config.texture_inversion_embedding)

        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler.set_timesteps(config.num_inference_steps)
        self.pipe.scheduler = self.scheduler

        self.scheduler.alphas = self.scheduler.alphas.to(self.device)
        self.scheduler.betas = self.scheduler.betas.to(self.device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

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

    def clip_image_at_percentiles(image, lower_percentile, upper_percentile):
        """
        Clips the image at the given lower and upper percentiles.
        """
        # Flatten the image to compute percentiles
        flattened_image = image.flatten()

        # Compute the lower and upper bounds
        lower_bound = torch.quantile(flattened_image, lower_percentile)
        upper_bound = torch.quantile(flattened_image, upper_percentile)

        # Clip the image
        clipped_image = torch.clamp(image, lower_bound, upper_bound)

        return clipped_image

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

    def pds_timestep_sampling(
        self,
        batch_size,
        sample: float=None # Range (0, 1)
    ):
        self.scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = reversed(self.scheduler.timesteps)

        if sample is None:
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
        else:
            assert sample < 1 + 1e-6
            assert sample > 0 - 1e-6
            idx = torch.tensor(
                [int(len(timesteps) * sample)],
                dtype=torch.long,
                device="cpu"
            )

        t = timesteps[idx].cpu().long()
        t_prev = timesteps[idx - 1].cpu().long()

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
        t_project,
        t_edit,
        source_im = None,
        prompt=None,
        num_solve_steps=20,
        project_cfg_scale=15,
        src_cfg_scale=100,
        tgt_cfg_scale=100,
        project_noise_sample=None,
        project_from=None,
        thresholding: Optional[Literal['dynamic']] = None,
        dynamic_thresholding_cutoffs: Optional[Tuple[float, float]] = None,
        loss_coefficients: Union[Tuple[float, float], Literal['z']] = 'z',
        extra_tgt_prompts=', detailed high resolution, high quality, sharp',
        extra_src_prompts=', oversaturated, smooth, pixelated, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed',
        src_method: Literal["sdedit", "step"] = "step",
        project_x_loss: bool = True,
        project_eps_loss: bool = True,
        reduction="mean",
        return_dict=False,
        t_max=1,
        run_as_rgb=False
    ):
        
        assert project_eps_loss or (not loss_coefficients == 'z'), 'Using Z formulation, eps loss must use projected x0'

        device = self.device
        scheduler = self.scheduler
        if run_as_rgb:
            with torch.enable_grad():
                im = self.encode_image(im)
        else:
            im = im

        # process text.
        self.update_text_features(tgt_prompt=prompt + extra_tgt_prompts, src_prompt=prompt + extra_src_prompts)
        tgt_text_embedding = self.tgt_text_feature
        src_text_embedding = self.src_text_feature
        uncond_embedding = self.null_text_feature

        tgt_x0 = im
        batch_size = im.shape[0]
        project_from = im if project_from is None else project_from

        if source_im is not None:
            src_x0 = source_im
        elif src_method == "sdedit":
            src_x0 = self.run_sdedit(
                x0=project_from,
                tgt_prompt=prompt,
                num_inference_steps=num_solve_steps,
                skip=int((1 - t_project) * num_solve_steps),
                noise_sample=project_noise_sample,
            )
        elif src_method == "step":

            projection_timestep, _ = self.pds_timestep_sampling(batch_size, sample=t_project)
            src_x0, noise = self.run_single_step(
                x0=im,
                t = projection_timestep.item(),
                tgt_prompt=prompt,
            )

        t, t_prev = self.pds_timestep_sampling(batch_size, sample=t_edit)
        beta_t = scheduler.betas[t].to(device)
        alpha_bar_t = scheduler.alphas_cumprod[t].to(device)
        alpha_bar_t_prev = scheduler.alphas_cumprod[t_prev].to(device)
        sigma_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t) ** (0.5)

        t_bad = (t*15).clamp(1, t_max)
        # alpha_prod_t_bad = scheduler.alphas_cumprod[t].to(device)
        # w_eps = ((1 - alpha_prod_t_bad) / alpha_prod_t_bad) ** 0.5

        noise = torch.randn_like(tgt_x0)
        noise_t_prev = torch.randn_like(tgt_x0)

        zts = dict()
        for latent, cond_text_embedding, name, cfg_scale in zip(
            [tgt_x0, src_x0], [tgt_text_embedding, src_text_embedding], ["tgt", "src"], [tgt_cfg_scale, src_cfg_scale]
        ):
            x_x0 = latent if project_x_loss else tgt_x0
            eps_x0 = latent if project_eps_loss else tgt_x0
            # latents_noisy = scheduler.add_noise(curr_x0, noise, t)
            if name == "tgt":
                latents_noisy = scheduler.add_noise(eps_x0, noise, t)
            else:
                latents_noisy = scheduler.add_noise(eps_x0, noise, t_bad)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            text_embeddings = torch.cat([cond_text_embedding, uncond_embedding], dim=0)
            noise_pred = self.unet.forward(
                latent_model_input,
                torch.cat([t] * 2).to(device),
                encoder_hidden_states=text_embeddings,
            ).sample
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

            scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(latents_noisy)
            latent_clean_pred = scheduler.step(noise_pred, t, latents_noisy).pred_original_sample

            if thresholding == 'dynamic':
                clean_pred = self.vae.decode(latent_clean_pred / 0.18215).sample
                clean_pred = (clean_pred / 2 + 0.5)
                low_clip, high_clip = dynamic_thresholding_cutoffs
                clean_pred = clip_image_at_percentiles(clean_pred, low_clip, high_clip)
                clean_pred -= clean_pred.min()
                clean_pred /= clean_pred.max()
                latent_clean_pred = self.encode_image(clean_pred)

            alpha_prod_t = scheduler.alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            noise_pred = (latents_noisy - latent_clean_pred * (alpha_prod_t ** (0.5))) / (beta_prod_t ** (0.5))

            if loss_coefficients == 'z':
                x_t_prev = scheduler.add_noise(latent, noise_t_prev, t_prev)
                mu = self.compute_posterior_mean(latents_noisy, noise_pred, t, t_prev)
                zt = (x_t_prev - mu) / sigma_t
            else:
                x_coeff, eps_coeff = loss_coefficients
                zt = x_coeff * x_x0 + eps_coeff * noise_pred
                # zt = x_coeff * x_x0 + eps_coeff * noise_pred * w_eps
            zts[name] = zt

        grad = zts["tgt"] - zts["src"]
        grad = torch.nan_to_num(grad)
        target = (tgt_x0 - grad).detach()
        loss = 0.5 * F.mse_loss(tgt_x0, target, reduction=reduction) / batch_size

        if return_dict:
            dic = {"loss": loss, "grad": grad, "t": t, "src_x0": src_x0, "target": target}
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
        self.update_text_features(tgt_prompt=prompt + ', detailed high resolution, high quality, sharp', src_prompt=prompt + ', oversaturated, smooth, pixelated, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed')
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
            # lower_bound =  999 - int(skip_percentage * 1000)
            lower_bound = 30
            upper_bound = 1000 - int(skip_percentage * 970)
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

            scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(latents_noisy)
            latent_clean_pred = scheduler.step(noise_pred, t, latents_noisy).pred_original_sample

            clean_pred = self.vae.decode(latent_clean_pred / 0.18215).sample
            clean_pred = (clean_pred / 2 + 0.5)
            clean_pred = clip_image_at_percentiles(clean_pred, 0.03, 0.97)
            clean_pred -= clean_pred.min()
            clean_pred /= clean_pred.max()
            clipped_latent_clean = self.encode_image(clean_pred)

            alpha_prod_t = scheduler.alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            noise_pred = (latents_noisy - clipped_latent_clean * (alpha_prod_t ** (0.5))) / (beta_prod_t ** (0.5))

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

    def reverse_process_loss(
        self,
        im,
        t_sample=None,
        prompt=None,
        reduction="mean",
        return_dict=False,
        latents_noisy=None,
        noise=None,
    ):
        device = self.device
        scheduler = self.scheduler

        # process text.
        self.update_text_features(tgt_prompt=prompt, src_prompt=prompt)
        tgt_text_embedding = self.tgt_text_feature
        src_text_embedding = self.src_text_feature
        uncond_embedding = self.null_text_feature

        batch_size = im.shape[0]
        t, t_prev = self.pds_timestep_sampling(batch_size, sample=t_sample)

        if noise is None:
            noise = torch.randn_like(im)

        # x_t
        if latents_noisy is None:
            latents_noisy = scheduler.add_noise(im, noise, t)
        latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
        text_embeddings = torch.cat([src_text_embedding, uncond_embedding], dim=0)
        noise_pred = self.unet.forward(
            latent_model_input,
            torch.cat([t] * 2).to(device),
            encoder_hidden_states=text_embeddings,
        ).sample
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_uncond)

        x_0_hat_pred = scheduler.step(noise_pred, t, latents_noisy).pred_original_sample

        # x_{t-1}
        latents_noisy_prev = scheduler.step(noise_pred, t, latents_noisy, eta=0).prev_sample
        latent_model_input = torch.cat([latents_noisy_prev] * 2, dim=0)
        text_embeddings = torch.cat([tgt_text_embedding, uncond_embedding], dim=0)
        noise_pred_prev = self.unet.forward(
            latent_model_input,
            torch.cat([t_prev] * 2).to(device),
            encoder_hidden_states=text_embeddings,
        ).sample
        noise_pred_text, noise_pred_uncond = noise_pred_prev.chunk(2)
        noise_pred_prev = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_uncond)

        alpha_prod_t_prev = scheduler.alphas_cumprod[t_prev].to(device)
        w = ((1 - alpha_prod_t_prev) / alpha_prod_t_prev) ** 0.5

        grad = w * (noise_pred_prev - noise_pred)
        grad = torch.nan_to_num(grad)
        target = (im - grad).detach()
        loss = 0.5 * F.mse_loss(im, target, reduction=reduction) / batch_size
        if return_dict:
            dic = {"loss": loss, "grad": grad, "t": t, "target": target, "noise_pred": noise_pred, "noise_pred_prev": noise_pred_prev, "latents_noisy_prev": latents_noisy_prev, "x_0_hat_pred": x_0_hat_pred}
            return dic
        else:
            return loss

    def run_sdedit(self, x0, tgt_prompt=None, num_inference_steps=20, skip=7, eta=0, noise_sample=None):
        scheduler = self.scheduler
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps
        reversed_timesteps = reversed(scheduler.timesteps)

        S = num_inference_steps - skip
        t = reversed_timesteps[S - 1]
        noise = torch.randn_like(x0) if noise_sample is None else noise_sample

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

        t = torch.tensor(t, device=x0.device, dtype=torch.long)
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

        return xhat_pred, noise

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
