from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import random
import os


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


save_path = 'results/coco_base'
seed = 1
pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16).to("cuda")
seed_everything(seed)
image = pipeline("A car in the side of a street", num_inference_steps=50).images[0]
image.save("%s/clean_%d.png" % (save_path, seed))

for iteration in range(500, 3000, 500):
    pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16).to("cuda")
    pipeline.load_textual_inversion("textual_inversion_pds/learned_embeds-steps-%s.safetensors" % iteration)
    seed_everything(seed)
    image = pipeline("A car in the side of a street. <pds>", num_inference_steps=50).images[0]
    image.save("%s/ti_%d_%d.png" % (save_path, iteration, seed))




