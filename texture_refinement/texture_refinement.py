
import os
import torch
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as optim
import wandb
import tyro

from dataclasses import dataclass, asdict, is_dataclass
from typing import Literal, Optional, Tuple, Any, Union

import matplotlib.pyplot as plt

import numpy as np
import math
from tqdm import tqdm

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftPhongShader,
)

import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to make the other directory importable
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from pds.pds import PDS, PDSConfig

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
class TextureRefinementConfig():
    wandb_enabled: bool = False
    experiment_name: Optional[str] = None
    seed: int = 0
    lr: float = 0.01
    n_steps: int = 10000
    num_views: int = 50
    num_views_per_iteration: int = 4
    render_resolution: int = 512
    camera_distance_range: Tuple[float, float] = (1.5, 3.0)
    camera_azimuth_range: Tuple[float, float] = (-180.0, 180.0)
    camera_elevation_range: Tuple[float, float] = (-10.0, 30.0)
    save_period: int = 100
    mesh_filename: str = 'cow_mesh/cow.obj'
    noise_texture: bool = False
    prompt: str = ''
    loss_coefficients: Union[Tuple[float, float], Literal['z']] = (0, 1)
    project_cfg_scale: float = 15
    extra_tgt_prompts: str = ', detailed high resolution, high quality, sharp'
    extra_src_prompts: str = '. <pds>'
    # extra_src_prompts: str = ', oversaturated, smooth, pixelated, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed'
    src_method: Literal['step', 'sdedit'] = 'step'
    pds_t_schedule: TimestepScheduleConfig = TimestepScheduleConfig(
        mode='fixed',
        upper_bound = 0.98,
        lower_bound = 0.02,
    )
    project_t_schedule: TimestepScheduleConfig = TimestepScheduleConfig(
        mode = 'fixed',
        upper_bound = 0.1,
        lower_bound = 0.02,
    )

def dataclass_to_dict(obj: Any) -> dict:
    if is_dataclass(obj):
        return {k: dataclass_to_dict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj

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

def permute_decoded_latent(decoded):
    rgb = decoded.detach()
    rgb = rgb.float().cpu().permute(0, 2, 3, 1)
    rgb = rgb.permute(1, 0, 2, 3)
    rgb = rgb.flatten(start_dim=1, end_dim=2)
    return rgb

def gaussian_kernel(size, sigma):
    """
    Creates a Gaussian Kernel with the given size and sigma
    """
    # Create a tensor with coordinates of a grid
    x = torch.arange(size).float() - size // 2
    y = torch.arange(size).float() - size // 2
    y, x = torch.meshgrid(y, x)

    # Calculate the gaussian kernel
    gaussian_kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Normalize the kernel so the sum is 1
    gaussian_kernel /= gaussian_kernel.sum()

    return gaussian_kernel.unsqueeze(0).unsqueeze(0)

def gaussian_blur(image, kernel_size, sigma):
    """
    Applies Gaussian Blur to the given image using the specified kernel size and sigma
    """

    kernel = gaussian_kernel(kernel_size, sigma).to(image)
    kernel = kernel.repeat(image.size(1), 1, 1, 1)
    padding = kernel_size // 2
    blurred_image = F.conv2d(image, kernel, padding=padding, groups=image.size(1))

    return blurred_image

def training_loop(config: TextureRefinementConfig, save_dir: str):

    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set paths
    DATA_DIR = "./data"
    obj_filename = os.path.join(DATA_DIR, config.mesh_filename)

    # Load obj file
    mesh = load_objs_as_meshes([obj_filename], device=device)

    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)));

    num_views = config.num_views

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    R, T = look_at_view_transform(dist=2, elev=0, azim=0)
    camera = FoVPerspectiveCameras(device=device, R=R[None, 0, ...],
                                    T=T[None, 0, ...])

    raster_settings = RasterizationSettings(
        image_size=config.render_resolution,
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    sigma = 1e-4
    raster_settings_soft = RasterizationSettings(
        image_size=512,
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
        faces_per_pixel=50,
        perspective_correct=False,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings,
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=camera,
            lights=lights
        )
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings_soft,
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=camera,
            lights=lights
        )
    )

    # Show a visualization comparing the rendered predicted mesh to the ground truth mesh
    def visualize_prediction(
        predicted_mesh,
        comparison_mesh,
        renderer=renderer, 
        # target_image=target_rgb[1],
        title='',
        save_dir='./results/',
        cameras=None,
    ):
        inds = range(3)
        with torch.no_grad():
            predicted_images = renderer(predicted_mesh) if cameras is None else renderer(predicted_mesh, cameras=cameras)
            target_image = renderer(comparison_mesh) if cameras is None else renderer(comparison_mesh, cameras=cameras)
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(predicted_images[0, ..., inds].cpu().detach().numpy())

        plt.subplot(1, 2, 2)
        plt.imshow(target_image[0, ..., inds].cpu().detach().numpy())
        plt.title(title)
        plt.axis("off")

        save_path = os.path.join(save_dir, f'save_{title}.png')
        plt.savefig(save_path, format='png')
        plt.clf()


    # Initialize PDS critic
    pds = PDS(PDSConfig(
        # sd_pretrained_model_or_path='runwayml/stable-diffusion-v1-5',
        sd_pretrained_model_or_path='stabilityai/stable-diffusion-2-1',
        texture_inversion_embedding='../pds/assets/learned_embeds-steps-1500.safetensors'
    ))

    # Initialize mesh with copy of cow mesh and its blurred texture
    opt_mesh = mesh.clone()
    texture_images = opt_mesh.textures.maps_padded()

    kernel_size = 15  # Size of the Gaussian kernel
    sigma = 1 # Standard deviation of the Gaussian distribution
    blurred_image = gaussian_blur(texture_images.permute(0, 3, 1, 2), kernel_size, sigma).permute(0, 2, 3, 1)
    if config.noise_texture:
        blurred_image += 0.3 * torch.randn_like(blurred_image)
        blurred_image = blurred_image.clamp(0, 1)

    # Convert the texture images to a Parameter for optimization
    opt_mesh.textures._maps_padded = Parameter(texture_images)
    opt_mesh.textures._maps_padded = Parameter(blurred_image)

    optimizer = optim.Adam([opt_mesh.textures.maps_padded()], lr=config.lr)

    num_views_per_iteration = config.num_views_per_iteration
    loop = tqdm(range(config.n_steps))

    losses = {"rgb": {"weight": 1.0, "values": []}}

    for step in loop:
        optimizer.zero_grad()
        
        # Losses to smooth /regularize the mesh shape
        loss = {k: torch.tensor(0.0, device=device) for k in losses}
        
        # Randomly select num_views_per_iteration views to optimize over in this iteration.
        # for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
        for _ in range(num_views_per_iteration):

            sample = torch.rand((1,))
            elev_min, elev_max = config.camera_elevation_range
            elev = elev_min + sample * (elev_max - elev_min)

            sample = torch.rand((1,))
            azim_min, azim_max = config.camera_azimuth_range
            azim = azim_min + sample * (azim_max - azim_min)

            sample = torch.rand((1,))
            dist_min, dist_max = config.camera_distance_range
            dist = dist_min + sample * (dist_max - dist_min)

            lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

            R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
            sample_cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

            images_predicted = renderer(opt_mesh, cameras=sample_cameras[0], lights=lights)
            
            predicted_rgb = images_predicted[..., :3].clamp(0, 1)
            latent_render = pds.encode_image(predicted_rgb.permute(0, 3, 1, 2))

            with torch.no_grad():
                pds_dict = pds.pds_gen(
                    im=latent_render,
                    t_project = sample_timestep(config.project_t_schedule, step),
                    t_edit = torch.rand(1).item() * 0.96 + 0.02,
                    prompt=config.prompt,
                    project_cfg_scale=config.project_cfg_scale,
                    extra_src_prompts=config.extra_src_prompts,
                    extra_tgt_prompts=config.extra_tgt_prompts,
                    thresholding=None,
                    loss_coefficients=config.loss_coefficients,
                    src_method=config.src_method,
                    return_dict=True
                )

            grad = pds_dict['grad']
            loss = pds_dict['loss']
            latent_render.backward(gradient=grad, retain_graph=True)

            # Throw away VAE grad to save memory
            for param in pds.vae.parameters():
                param.grad = None
        
        # Print the losses
        loop.set_description("loss = %.6f" % loss)
        
        # Plot mesh
        if step % config.save_period == 0:
            for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
                visualize_prediction(
                    opt_mesh,
                    comparison_mesh=mesh,
                    renderer=renderer,
                    title=f'iter_{step}',
                    cameras=sample_cameras[0],
                    # target_image=target_rgb[j],
                    save_dir=save_dir,
                )
            
        # Optimization step
        optimizer.step()


def run_texture_refinement(config: TextureRefinementConfig):

    config_dict = dataclass_to_dict(config)
    experiment_name = 'texture_refinement_%s/%s_lr%.3f_seed_%d' % (
        config.src_method,
        config.prompt.replace(' ', '_') if len(config.prompt) > 0 else 'no_prompt',
        config.lr,
        config.seed,
    ) if config.experiment_name is None else config.experiment_name

    if config.wandb_enabled:
        wandb.init(
            project="Texture-Refinement",
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
    tyro.cli(run_texture_refinement)