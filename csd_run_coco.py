import wandb
import torch
import torch.nn.functional as F
import numpy as np
import math
import tyro
from dataclasses import dataclass, asdict, is_dataclass
import matplotlib.pyplot as plt
import imageio
from typing import Literal, Optional, Tuple, Any, Union
from pds.utils.imageutil import permute_decoded_latent
from pds.utils.trainutil import seed_everything
import os
from tqdm import tqdm
from PIL import Image
from datetime import datetime

from pds.pds import PDS, PDSConfig
# from pds.pds_sdxl import PDS_sdxl, PDS_sdxlConfig
from pds_train.coco_utils import CocoDataset

@dataclass
class SimpleScheduleConfig():
    mode: Literal['fixed', 'schedule'] = 'fixed'
    value_initial: float = 0.98
    value_final: float = 0.03
    warmup_steps: Optional[int] = None
    num_steps: Optional[int] = None

@dataclass
class TimestepScheduleConfig():
    mode: Literal['fixed', 'schedule'] = 'fixed'
    schedule: Optional[Literal['linear', 'cosine']] = None
    upper_bound: float = 0.98
    lower_bound: float = 0.03
    upper_bound_final: Optional[float] = None
    lower_bound_final: Optional[float] = None
    warmup_steps: Optional[int] = None
    num_steps: Optional[int] = None

@dataclass
class PDSGenerationConfig():
    wandb_enabled: bool = False
    experiment_name: Optional[str] = None
    # lr: float = 0.008
    lr: SimpleScheduleConfig = SimpleScheduleConfig(
        mode='schedule',
        # value_initial=0.01,
        value_initial=0.01,
        value_final=0.01,
        warmup_steps=500,
        num_steps=501,
    )
    loss_coefficients: Union[Tuple[float, float], Literal['z']] = (0.2, 1) # Set coefficients for x and eps terms, alternatively use z weighting
    x_coeff: SimpleScheduleConfig = SimpleScheduleConfig(
        mode='schedule',
        value_initial=1,
        value_final=1,
        warmup_steps=500,
        num_steps=501,
    )
    eps_coeff: SimpleScheduleConfig = SimpleScheduleConfig(
        mode='schedule',
        value_initial=1,
        value_final=0,
        warmup_steps=500,
        num_steps=501,
    )
    n_steps: int = 2800
    seed: int = 48
    optimize_canvas: bool = False
    fixed_noise_sample: bool = False
    project_from_original: bool = False
    model: Literal['sd', 'sdxl'] = 'sd'
    src_method: Literal['step', 'sdedit'] = 'step'
    thresholding: Optional[Literal['dynamic']] = None
    dynamic_thresholding_cutoffs: Optional[Tuple[float, float]] = (0.015, 0.985)
    prompt: str = 'a DSLR photo of a dog in a winter wonderland'
    extra_tgt_prompts: str = ''
    extra_src_prompts: str = ', oversaturated, smooth, pixelated, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed'
    # extra_src_prompts: str = '. <pds>'
    load_lora: bool = False
    init_image_fn: Optional[str] = None
    # project_cfg: float = 15
    project_cfg: SimpleScheduleConfig = SimpleScheduleConfig(
        mode='schedule',
        value_initial=100,
        # value_initial=40,
        value_final=7.5,
        warmup_steps=500,
        num_steps=501,
    )
    pds_cfg: float = 100
    project_x_loss: bool = True
    project_eps_loss: bool = False
    pds_t_schedule: TimestepScheduleConfig = TimestepScheduleConfig(
        mode='fixed',
        upper_bound = 0.98,
        lower_bound = 0.02,
    )

    project_t_schedule: TimestepScheduleConfig = TimestepScheduleConfig(
        mode='fixed',
        upper_bound = 0.98,
        lower_bound = 0.02,
    )
    shard_id: int = 0

def validate_timestep_config(cfg: TimestepScheduleConfig):
    if cfg.mode == 'fixed':
        if not (
            cfg.upper_bound_final is None and 
            cfg.lower_bound_final is None and 
            cfg.num_steps is None and 
            cfg.warmup_steps is None
        ):
            raise ValueError('Only set upper_bound and lower_bound for fixed timestep schedule.')
    if cfg.mode == 'schedule':
        if (
            cfg.upper_bound_final is None or 
            cfg.lower_bound_final is None or 
            cfg.num_steps is None or 
            cfg.warmup_steps is None or
            cfg.schedule is None
        ):
            raise ValueError('Set all config parameters for timestep schedule.')

def dataclass_to_dict(obj: Any) -> dict:
    if is_dataclass(obj):
        return {k: dataclass_to_dict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj
    
def sample_simple_schedule(sched: SimpleScheduleConfig, train_step: int):
    interp = (train_step - sched.warmup_steps) / (sched.num_steps - sched.warmup_steps)
    interp = max(0, interp)
    interp = min(1, interp)

    sample = sched.value_initial + interp * (sched.value_final - sched.value_initial)
    return sample


# Returns timestep normalized to range (0, 1)
def sample_timestep(t_cfg: TimestepScheduleConfig, train_step: int):

    random_float = torch.rand(size=(1,)).item()

    if t_cfg.mode == 'fixed':
        return t_cfg.lower_bound + (t_cfg.upper_bound - t_cfg.lower_bound) * random_float
    
    interp = (train_step - t_cfg.warmup_steps) / (t_cfg.num_steps - t_cfg.warmup_steps)
    interp = max(0, interp)
    interp = min(1, interp)

    if t_cfg.schedule == 'cosine':
        interp = (1 - math.cos(interp * math.pi)) / 2
    else:
        assert t_cfg.schedule == 'linear'

    lower_bound = t_cfg.lower_bound + interp * (t_cfg.lower_bound_final - t_cfg.lower_bound)
    upper_bound = t_cfg.upper_bound + interp * (t_cfg.upper_bound_final - t_cfg.upper_bound)

    return lower_bound + (upper_bound - lower_bound) * random_float


def training_loop(config: PDSGenerationConfig, save_dir: str):

    os.makedirs(save_dir, exist_ok=True)
    seed_everything(config.seed)

    if config.model == 'sd':
        pds = PDS(PDSConfig(
            sd_pretrained_model_or_path='stabilityai/stable-diffusion-2-1-base',
        ))
        latent_dim = 64
    
    dataset =  CocoDataset(root='/fs/vulcan-datasets/coco/images/train2017', json='/fs/vulcan-datasets/coco/annotations/captions_train2017.json', vocab=None)
    for i in range(config.shard_id*100, (config.shard_id+1)*100):
        print(i)
        ref_im, prompt, _ = dataset.__getitem__(i)

        if config.init_image_fn is not None:
            reference = torch.tensor(plt.imread(config.init_image_fn))[..., :3]
            reference = reference.permute(2, 0, 1)[None, ...]
            reference = reference.to(pds.unet.device)

            reference_latent = pds.encode_image(reference)
            im = reference_latent.clone()
        elif not config.optimize_canvas:
            im = torch.randn((1, 4, latent_dim, latent_dim), device=pds.unet.device)
        else:
            im = torch.randn((1, 4, int(latent_dim * 1.3), int(1.3 * latent_dim)), device=pds.unet.device)

        noise_sample = torch.randn((1, 4, latent_dim, latent_dim), device=pds.unet.device)

        im.requires_grad_(True)
        im.retain_grad()

        im_original = im.clone()

        im_optimizer = torch.optim.AdamW([im], lr=config.lr.value_initial, betas=(0.9, 0.99), eps=1e-15)
        decodes = []
        decodes_src = []

        for step in tqdm(range(config.n_steps)):
            im_optimizer.zero_grad()

            if config.optimize_canvas:
                rand_float = torch.rand((1,)).item()
                scale_min, scale_max = 50, 70
                scale = int(scale_min + rand_float * (scale_max - scale_min))
                rand_float = torch.rand((1,)).item()
                x = (im.shape[3] - scale) * rand_float
                x = int(x)
                rand_float = torch.rand((1,)).item()
                y = (im.shape[2] - scale) * rand_float
                y = int(y)

                im_opt = im[:, :, y:y+scale, x:x+scale]
                im_opt = F.interpolate(im_opt, (latent_dim, latent_dim))
                im_og = im_original[:, :, y:y+scale, x:x+scale]
                im_og = F.interpolate(im_og, (latent_dim, latent_dim))
            else:
                im_opt = im
                im_og = im_original

            with torch.no_grad():

                x_coeff = sample_simple_schedule(config.x_coeff, step)
                eps_coeff = sample_simple_schedule(config.eps_coeff, step)

                pds_dict = pds.pds_gen(
                    im=im_opt,
                    t_project = sample_timestep(config.project_t_schedule, step),
                    t_edit = sample_timestep(config.pds_t_schedule, step),
                    prompt=prompt,
                    extra_src_prompts=config.extra_src_prompts,
                    extra_tgt_prompts=config.extra_tgt_prompts,
                    thresholding=config.thresholding,
                    dynamic_thresholding_cutoffs=config.dynamic_thresholding_cutoffs,
                    loss_coefficients=(x_coeff, eps_coeff),
                    project_x_loss=config.project_x_loss,
                    project_eps_loss=config.project_eps_loss,
                    project_cfg_scale=100,
                    src_cfg_scale=config.pds_cfg,
                    tgt_cfg_scale=config.pds_cfg,
                    project_noise_sample=noise_sample if config.fixed_noise_sample else None,
                    project_from=im_og if config.project_from_original else None,
                    src_method=config.src_method,
                    return_dict=True,
                    run_csd=True,
                )

            grad = pds_dict['grad']

            im_opt.backward(gradient=grad)

            im_optimizer.step()

            if (step+1) % 100 == 0:
                with torch.no_grad():
                    decoded = permute_decoded_latent(pds.decode_latent(im.detach())).cpu().numpy()
                decodes.append(decoded)
                prompt_print = prompt.replace(' ', '_')
                plt.imsave(os.path.join(save_dir ,f'{prompt_print}_ours.jpg'), decoded)

            current_lr = sample_simple_schedule(config.lr, step)
            for param_group in im_optimizer.param_groups:
                param_group['lr'] = current_lr


def run_pds_gen(config: PDSGenerationConfig):

    config_dict = dataclass_to_dict(config)
    date = datetime.now().strftime("%m-%d")
    experiment_name = 'eval/%s/%s_csd_gen_%s/%s_lr%.3f_seed%d' % (
        date,
        config.model,
        config.src_method,
        config.prompt.replace(' ', '_'),
        config.lr.value_initial,
        config.seed
    ) if config.experiment_name is None else config.experiment_name

    training_loop(
        config,
        save_dir = f'results/{experiment_name}'
    )


if __name__ == "__main__":
    tyro.cli(run_pds_gen)