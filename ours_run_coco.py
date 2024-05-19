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

from pds.pds_songwei import PDS, PDSConfig
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
    run_name: Optional[str] = 'xds'
    experiment_name: Optional[str] = None
    lr: float = 0.008
    loss_coefficients: Union[Tuple[float, float], Literal['z']] = (0.3, 1) # Set coefficients for x and eps terms, alternatively use z weighting
    n_steps: int = 3200
    seed: int = 45
    n_samples: int = 5
    optimize_canvas: bool = False
    fixed_noise_sample: bool = False
    project_from_original: bool = False
    model: Literal['sd', 'sdxl'] = 'sd'
    src_method: Literal['step', 'sdedit'] = 'step'
    thresholding: Optional[Literal['dynamic']] = None
    dynamic_thresholding_cutoffs: Optional[Tuple[float, float]] = (0.015, 0.985)
    prompt: str = 'a DSLR photo of a dog in a winter wonderland'
    extra_tgt_prompts: str = '.'
    extra_src_prompts: str = ', oversaturated, smooth, pixelated, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed'
    init_image_fn: Optional[str] = None
    pds_cfg: float = 100
    project_x_loss: bool = True
    project_eps_loss: bool = True
    init_x0: bool = False
    adaptive_src_prompt: bool = False
    pds_t_schedule: TimestepScheduleConfig = TimestepScheduleConfig(
        mode='fixed',
        upper_bound = 0.98,
        lower_bound = 0.02,
    )
    project_t_schedule: TimestepScheduleConfig = TimestepScheduleConfig(
        mode = 'schedule',
        schedule = 'linear',
        upper_bound = 0.98,
        lower_bound = 0.03,
        upper_bound_final = 0.05,
        lower_bound_final = 0.02,
        warmup_steps = 200,
        num_steps = 1600,
    )

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
    else:
        pds = PDS_sdxl(PDS_sdxlConfig(
            sd_pretrained_model_or_path="stabilityai/stable-diffusion-xl-base-1.0"
        ))
        latent_dim = 128
    
    dataset =  CocoDataset(root='/fs/vulcan-datasets/coco/images/train2017', json='/fs/vulcan-datasets/coco/annotations/captions_train2017.json', vocab=None)
    for i in range(config.n_samples):
        ref_im, prompt = dataset.__getitem__(i)

        if config.init_image_fn is not None:
            reference = torch.tensor(plt.imread(config.init_image_fn))[..., :3]
            reference = reference.permute(2, 0, 1)[None, ...]
            reference = reference.to(pds.unet.device)

            reference_latent = pds.encode_image(reference)
            im = reference_latent.clone()
        elif not config.optimize_canvas:
            im = torch.randn((1, 4, latent_dim, latent_dim), device=pds.unet.device)
        else:
            im = torch.randn((1, 4, int(latent_dim * 1.3), int(1.3 * latent_dim)), device=pds.unet.device) * 0

        noise_sample = torch.randn((1, 4, latent_dim, latent_dim), device=pds.unet.device)

        im.requires_grad_(True)
        im.retain_grad()

        im_original = im.clone()

        im_optimizer = torch.optim.AdamW([im], lr=config.lr, betas=(0.9, 0.99), eps=1e-15)
        decodes = []
        decodes_src = []

        src_prompts = prompt + config.extra_src_prompts

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

                pds_dict = pds.pds_gen(
                    im=im_opt,
                    t_project = sample_timestep(config.project_t_schedule, step),
                    t_edit = sample_timestep(config.pds_t_schedule, step),
                    prompt=prompt,
                    src_prompts=src_prompts,
                    extra_tgt_prompts=config.extra_tgt_prompts,
                    thresholding=config.thresholding,
                    dynamic_thresholding_cutoffs=config.dynamic_thresholding_cutoffs,
                    loss_coefficients=config.loss_coefficients,
                    project_x_loss=config.project_x_loss,
                    project_eps_loss=config.project_eps_loss,
                    project_cfg_scale=7.5,
                    src_cfg_scale=config.pds_cfg,
                    tgt_cfg_scale=config.pds_cfg,
                    project_noise_sample=noise_sample if config.fixed_noise_sample else None,
                    project_from=im_og if config.project_from_original else None,
                    src_method=config.src_method,
                    return_dict=True
                )

            grad = pds_dict['grad']
            src_x0 = pds_dict['src_x0']

            im_opt.backward(gradient=grad)

            im_optimizer.step()

            if step % 10 == 0:
                with torch.no_grad():
                    decoded = permute_decoded_latent(pds.decode_latent(im)).cpu().numpy()
                    decoded_src = permute_decoded_latent(pds.decode_latent(src_x0)).cpu().numpy()
                decodes.append(decoded)
                decodes_src.append(decoded_src)
                prompt_print = prompt.replace(' ', '_')
                plt.imsave(os.path.join(save_dir ,f'{prompt_print}_ours.jpg'), decoded)
                plt.imsave(os.path.join(save_dir ,f'{prompt_print}_gt.jpg'), ref_im)
                plt.imsave(os.path.join(save_dir ,'pds_gen_project_x0.jpg'), decoded_src)

                if config.wandb_enabled:
                    wandb.log({'Learning Rate': config.lr})
                    wandb.log({"Current Target (Optim.)": wandb.Image(decoded)})
                    wandb.log({"Target Projection (Proj. Good)": wandb.Image(decoded_src)})

            if step % 100 == 0:
                imageio.mimwrite(os.path.join(save_dir, 'pds_cat_tgt.mp4'), np.stack(decodes).astype(np.float32)*255, fps=10, codec='libx264')
                imageio.mimwrite(os.path.join(save_dir, 'pds_cat_x0.mp4'), np.stack(decodes_src).astype(np.float32)*255, fps=10, codec='libx264')


def run_pds_gen(config: PDSGenerationConfig):

    config_dict = dataclass_to_dict(config)
    date = datetime.now().strftime("%m-%d")
    experiment_name = 'eval/%s/%s_%s_%s/lr%.3f_seed%d_%s' % (
        date,
        config.run_name,
        config.model,
        config.src_method,
        config.lr,
        config.seed,
        'initx0' if config.init_x0 is not None else 'no_initx0'
    ) if config.experiment_name is None else config.experiment_name

    if config.wandb_enabled:
        wandb.init(
            project="PDS-Gen",
            name=experiment_name,
            config=config_dict
        )

    training_loop(
        config,
        save_dir = f'results/{experiment_name}'
    )

    if config.wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    tyro.cli(run_pds_gen)
