from collections import defaultdict
import os
from pathlib import Path
import random
from dataclasses import dataclass, field
from typing import Literal, Optional, Type, Union

import torch
from itertools import cycle
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from torch.cuda.amp.grad_scaler import GradScaler

from pds_nerf.modified_nerfstudio.base_pipeline import \
    ModifiedVanillaPipeline
from pds_nerf.pds_datamanager import PDSDataManagerConfig
from pds.pds import PDS, PDSConfig, tensor_to_pil, pil_to_tensor
from pds.utils import imageutil
from nerfstudio.utils import profiler


@dataclass
class RefinementPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: RefinementPipeline)

    datamanager: PDSDataManagerConfig = PDSDataManagerConfig()
    pds: PDSConfig = PDSConfig()
    pds_device: Optional[Union[torch.device, str]] = None

    skip_min_ratio: float = 0.8
    skip_max_ratio: float = 0.9

    log_step: int = 100
    edit_rate: int = 10
    edit_count: int = 1


class RefinementPipeline(ModifiedVanillaPipeline):
    config: RefinementPipelineConfig

    def __init__(
        self,
        config: RefinementPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
        **kwargs,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler, **kwargs)

        # Construct PDS
        self.pds_device = (
            torch.device(device) if self.config.pds_device is None else torch.device(self.config.pds_device)
        )
        self.config.pds.device = self.pds_device
        self.pds = PDS(self.config.pds)

        if self.datamanager.config.train_num_images_to_sample_from == -1:
            self.train_indices_order = cycle(range(len(self.datamanager.train_dataparser_outputs.image_filenames)))
        else:
            self.train_indices_order = cycle(range(self.datamanager.config.train_num_images_to_sample_from))

    def get_train_loss_dict(self, step: int):
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        if step % self.config.edit_rate == 0:
            for i in range(self.config.edit_count):
                current_spot = next(self.train_indices_order)
                original_image = self.datamanager.original_image_batch["image"][current_spot].to(self.device)
                current_index = self.datamanager.image_batch["image_idx"][current_spot]

                camera_transforms = self.datamanager.train_camera_optimizer(current_index.unsqueeze(dim=0))
                current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index].to(self.device)
                current_ray_bundle = current_camera.generate_rays(
                    torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms
                )
                original_image = original_image.unsqueeze(dim=0).permute(0, 3, 1, 2)  # [B, 3, H, W]
                camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)
                rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)  # [B, 3, H, W]

                # delete to free up memory
                del camera_outputs
                del current_camera
                del current_ray_bundle
                del camera_transforms
                torch.cuda.empty_cache()

                input_img = original_image

                with torch.no_grad():
                    h, w = input_img.shape[2:]
                    l = min(h, w)
                    h = int(h * 512 / l)
                    w = int(w * 512 / l)

                    resized_img = torch.nn.functional.interpolate(input_img, size=(h, w), mode="bilinear")
                    latents = self.pds.encode_image(resized_img.to(self.pds_device))

                ## config ##
                x0 = latents
                num_inference_steps = self.pds.config.num_inference_steps
                min_step = int(num_inference_steps * self.config.skip_min_ratio)
                max_step = int(num_inference_steps * self.config.skip_max_ratio)
                skip = random.randint(min_step, max_step)

                edit_x0 = self.pds.run_sdedit(x0, skip=skip)
                edit_img = self.pds.decode_latent(edit_x0)

                if edit_img.size() != rendered_image.size():
                    edit_img = torch.nn.functional.interpolate(
                        edit_img, size=rendered_image.size()[2:], mode="bilinear"
                    )

                self.datamanager.image_batch["image"][current_spot] = edit_img.squeeze().permute(1, 2, 0)  # [H,W,3]

            if step % self.config.log_step == 0:
                with torch.no_grad():
                    rendered_img_pil = tensor_to_pil(rendered_image)
                    edit_img_pil = tensor_to_pil(edit_img)
                    rw, rh = float("inf"), float("inf")
                    for img in [rendered_img_pil, edit_img_pil]:
                        w, h = img.size
                        rw = min(w, rw)
                        rh = min(h, rh)
                    rendered_img_pil = rendered_img_pil.resize((rw, rh))
                    edit_img_pil = edit_img_pil.resize((rw, rh))
                    save_img_pil = imageutil.merge_images([rendered_img_pil, edit_img_pil])
                    save_img_pil.save(self.base_dir / f"logging/replace-out-{step}.png")

        kwargs = {"use_rgb_loss": True}
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict, **kwargs)

        return model_outputs, loss_dict, metrics_dict

