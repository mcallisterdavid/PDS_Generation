from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.linalg import norm
from torch import FloatTensor, LongTensor, Tensor, Size, lerp, zeros_like
from diffusers import DDIMScheduler, DiffusionPipeline
from jaxtyping import Float
from typing import Literal, Optional, Tuple, Union
from PIL import Image
from copy import deepcopy
import matplotlib.pyplot as plt

@dataclass
class PDSConfig:
    sd_pretrained_model_or_path: str = "runwayml/stable-diffusion-v1-5"
    texture_inversion_embedding: str = "./assets/learned_embeds-steps-1500.safetensors"
    lora_weights: str = "./pds_train/sd-model-lora-same-noise-cfg-50-augmentations/checkpoint-1500/"

    # num_inference_steps: int = 500
    num_inference_steps: int = 50
    min_step_ratio: float = 0.02
    max_step_ratio: float = 0.98

    src_prompt: str = "a photo of a man"
    tgt_prompt: str = "a photo of a man raising his arms"

    guidance_scale: float = 100
    sdedit_guidance_scale: float = 15
    device: torch.device = torch.device("cuda")

    load_lora: bool = False


class PDS(object):
    def __init__(self, config: PDSConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.pipe = DiffusionPipeline.from_pretrained(config.sd_pretrained_model_or_path).to(self.device)
        try:
            self.pipe.load_textual_inversion(config.texture_inversion_embedding)
        except:
            print("UNABLE TO LOAD TEXTUAL INVERSION TOKENS")

        if config.load_lora:
            print("LOADING LORA ADAPTER")
            pipe_lora = deepcopy(self.pipe)
            # pipe_lora.load_lora_weights("./pds_train/sd-model-finetuned-lora/", weight_name="pytorch_lora_weights.safetensors")
            # pipe_lora.load_lora_weights("./pds_train/sd-model-lora-same-noise/checkpoint-2000/", weight_name="pytorch_lora_weights.safetensors")
            # pipe_lora.load_lora_weights("./pds_train/sd-model-lora-same-noise-cfg-50/checkpoint-1500/", weight_name="pytorch_lora_weights.safetensors")
            # pipe_lora.load_lora_weights("./pds_train/sd-model-lora-same-noise-cfg-50-augmentations/checkpoint-1500/", weight_name="pytorch_lora_weights.safetensors")
            # pipe_lora.load_lora_weights("./pds_train/sd-model-lora-same-noise-low-cfg/", weight_name="pytorch_lora_weights.safetensors")
            # pipe_lora.load_lora_weights("/home/mcallisterdavid/PDS_Generation/pds_train/sd-model-lora-same-noise-low-cfg-high-strength/checkpoint-2000/", weight_name="pytorch_lora_weights.safetensors")
            # pipe_lora.load_lora_weights("/home/mcallisterdavid/PDS_Generation/pds_train/sd-model-may-11-1/checkpoint-1000/", weight_name="pytorch_lora_weights.safetensors")
            # pipe_lora.load_lora_weights("/home/mcallisterdavid/PDS_Generation/pds_train/sd-model-may-11-6/checkpoint-500/", weight_name="pytorch_lora_weights.safetensors")
            # pipe_lora.load_lora_weights("/home/mcallisterdavid/PDS_Generation/pds_train/sd-model-may-11-4/checkpoint-2000/", weight_name="pytorch_lora_weights.safetensors")
            # pipe_lora.load_lora_weights("/home/mcallisterdavid/PDS_Generation/pds_train/sd-model-may-11-8/checkpoint-500/", weight_name="pytorch_lora_weights.safetensors")
            # pipe_lora.load_lora_weights("/home/mcallisterdavid/PDS_Generation/pds_train/sd-model-may-11-9/checkpoint-1000/", weight_name="pytorch_lora_weights.safetensors")
            # pipe_lora.load_lora_weights(config.lora_weights, weight_name="pytorch_lora_weights.safetensors")
            self.lora_unet = pipe_lora.unet
            del pipe_lora

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
        project_eps_loss: bool = False,
        reduction="mean",
        return_dict=False,
    ):
        
        # Temporary, not projecting is faster
        assert not project_eps_loss
        
        assert project_eps_loss or (not loss_coefficients == 'z'), 'Using Z formulation, eps loss must use projected x0'

        device = self.device
        scheduler = self.scheduler

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
            src_x0 = self.run_single_step(
                x0=project_from,
                t=projection_timestep.item(),
                tgt_prompt=prompt,
                cfg_scale=project_cfg_scale,
                noise_sample=project_noise_sample,
            )

        t, t_prev = self.pds_timestep_sampling(batch_size, sample=t_edit)
        beta_t = scheduler.betas[t].to(device)
        alpha_bar_t = scheduler.alphas_cumprod[t].to(device)
        alpha_bar_t_prev = scheduler.alphas_cumprod[t_prev].to(device)
        sigma_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t) ** (0.5)

        noise = torch.randn_like(tgt_x0)
        noise_t_prev = torch.randn_like(tgt_x0)

        unets = (self.unet, self.lora_unet) if self.config.load_lora else (self.unet, self.unet)

        zts = dict()
        for unet, latent, cond_text_embedding, name, cfg_scale in zip(
            unets, [tgt_x0, src_x0], [tgt_text_embedding, src_text_embedding], ["tgt", "src"], [tgt_cfg_scale, src_cfg_scale]
        ):
            x_x0 = latent if project_x_loss else tgt_x0
            eps_x0 = latent if project_eps_loss else tgt_x0
            latents_noisy = scheduler.add_noise(eps_x0, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 1, dim=0)
            # text_embeddings = torch.cat([cond_text_embedding, uncond_embedding], dim=0)
            text_embeddings = cond_text_embedding
            noise_pred = unet.forward(
                latent_model_input,
                torch.cat([t] * 1).to(device),
                encoder_hidden_states=text_embeddings,
            ).sample
            # noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            # noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

            scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(latents_noisy)
            latent_clean_pred = scheduler.step(noise_pred, t, latents_noisy).pred_original_sample

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
        

    def slerp(self, v0: FloatTensor, v1: FloatTensor, t, DOT_THRESHOLD=0.9995):
        '''
        Spherical linear interpolation
        Args:
            v0: Starting vector
            v1: Final vector
            t: Float value between 0.0 and 1.0
            DOT_THRESHOLD: Threshold for considering the two vectors as
                                    colinear. Not recommended to alter this.
        Returns:
            Interpolation vector between v0 and v1
        '''
        assert v0.shape == v1.shape, "shapes of v0 and v1 must match"

        # Normalize the vectors to get the directions and angles
        v0_norm: FloatTensor = norm(v0, dim=-1)
        v1_norm: FloatTensor = norm(v1, dim=-1)

        v0_normed: FloatTensor = v0 / v0_norm.unsqueeze(-1)
        v1_normed: FloatTensor = v1 / v1_norm.unsqueeze(-1)

        # Dot product with the normalized vectors
        dot: FloatTensor = (v0_normed * v1_normed).sum(-1)
        dot_mag: FloatTensor = dot.abs()

        # if dp is NaN, it's because the v0 or v1 row was filled with 0s
        # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
        gotta_lerp: LongTensor = dot_mag.isnan() | (dot_mag > DOT_THRESHOLD)
        can_slerp: LongTensor = ~gotta_lerp

        t_batch_dim_count: int = max(0, t.dim()-v0.dim()) if isinstance(t, Tensor) else 0
        t_batch_dims: Size = t.shape[:t_batch_dim_count] if isinstance(t, Tensor) else Size([])
        out: FloatTensor = zeros_like(v0.expand(*t_batch_dims, *[-1]*v0.dim()))

        # if no elements are lerpable, our vectors become 0-dimensional, preventing broadcasting
        if gotta_lerp.any():
            lerped: FloatTensor = lerp(v0, v1, t)

            out: FloatTensor = lerped.where(gotta_lerp.unsqueeze(-1), out)

        # if no elements are slerpable, our vectors become 0-dimensional, preventing broadcasting
        if can_slerp.any():

            # Calculate initial angle between v0 and v1
            theta_0: FloatTensor = dot.arccos().unsqueeze(-1)
            sin_theta_0: FloatTensor = theta_0.sin()
            # Angle at timestep t
            theta_t: FloatTensor = theta_0 * t
            sin_theta_t: FloatTensor = theta_t.sin()
            # Finish the slerp algorithm
            s0: FloatTensor = (theta_0 - theta_t).sin() / sin_theta_0
            s1: FloatTensor = sin_theta_t / sin_theta_0
            slerped: FloatTensor = s0 * v0 + s1 * v1

            out: FloatTensor = slerped.where(can_slerp.unsqueeze(-1), out)
        
        return out
    
    def analytical_flow_loss(
        self,
        im,
        prompt=None,
        cfg=7.5,
        alpha1=0.0,
        alpha2=1.0,
        t=None,
        reduction="mean",
        return_dict=False,
    ):
        device = self.device
        scheduler = self.scheduler

        self.update_text_features(tgt_prompt=prompt)
        tgt_text_embedding = self.tgt_text_feature
        uncond_embedding = self.null_text_feature

        batch_size = im.shape[0]
        t, _ = self.pds_timestep_sampling(batch_size)# if t is None else (t, None)

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
        noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)

        # alpha1_min, alpha1_max = 0.95, 0.999
        # interp = 1 - (t.item() / 1000)
        # alpha1 = alpha1_min + interp * (alpha1_max - alpha1_min)
        # alpha2 = 1 - alpha1

        alpha1 = 0.9

        # marginal_flow = self.slerp(noise_pred_text, noise, alpha1)
        marginal_flow = self.slerp(noise_pred, noise, alpha1)

        # marginal_flow = alpha1 * noise_pred_text + alpha2 * noise

        w = 1 - scheduler.alphas_cumprod[t].to(device)
        # grad = w * (noise_pred - noise)
        grad = w * (noise_pred - marginal_flow)
        grad = torch.nan_to_num(grad)
        target = (im - grad).detach()
        loss = 0.5 * F.mse_loss(im, target, reduction=reduction) / batch_size
        if return_dict:
            dic = {"loss": loss, "grad": grad, "t": t}
            return dic
        else:
            return loss
    
    def sds_loss(
        self,
        im,
        prompt=None,
        cfg=100,
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
        noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)

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

        noise = torch.randn_like(im)

        # x_t
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

        grad = w * (noise_pred - noise_pred_prev)
        grad = torch.nan_to_num(grad)
        target = (im - grad).detach()
        loss = 0.5 * F.mse_loss(im, target, reduction=reduction) / batch_size
        if return_dict:
            dic = {"loss": loss, "grad": grad, "t": t, "target": target}
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

    def run_single_step(
        self,
        x0,
        t: int,
        cfg_scale: float = 15,
        tgt_prompt=None,
        percentile_clip=False,
        eta=0,
        noise_sample=None
    ):
        scheduler = self.scheduler

        t = torch.tensor(t, device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0) if noise_sample is None else noise_sample

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
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

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
