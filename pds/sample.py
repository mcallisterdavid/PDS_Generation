from diffusers import DiffusionPipeline
import torch
import argparse
import imageio
import os
import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, default='A DSLR photo of a dog in a winter wonderland')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

seed_everything(args.seed)

outpath = '/nfshomes/songweig/hcl/code/PDS_Generation/results/samples'

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16)
pipeline.to("cuda")
img = pipeline(args.prompt).images[0]
fname = os.path.join(outpath, f"{args.prompt.replace(' ', '_')}_{args.seed}.jpg")
imageio.imwrite(fname, img)