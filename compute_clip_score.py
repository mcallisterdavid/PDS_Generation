import os
import torch
import clip
from PIL import Image
import torchvision
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/16", device=device)
# model, preprocess = clip.load("ViT-B/32", device=device)
model, preprocess = clip.load("ViT-L/14", device=device)

base_path = 'results/eval/05-19/vsd_gen__/lr0.010_seed0_scale7.5'
# base_path = 'results/eval/05-18/sds_gen__/lr0.100_seed0_scale100.0'


clip_scores = []
for fn in os.listdir(base_path):
    prompt = fn.split('_ours.jpg')[0].replace('_', ' ')
    text = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    image_path = f'{base_path}/{fn}'
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        clip_score = image_features @ text_features.T
    clip_scores.append(clip_score.item())
    # print(f'Clip score for id {len(clip_scores)} and prompt {prompt}: {clip_score}')

print('The average clip score and std for folder', base_path)
print('scores %.2f\pm%.2f'%(np.mean(clip_scores), np.std(clip_scores)))



# ViT-B/32
# The average clip score and std for method Ours
# Average: 0.30890963115389386
# The average clip score and std for method SDS
# Average: 0.3022119431268601
# The average clip score and std for method VSD
# Average: 0.29536588154141863

# ViT-L/14
# The average clip score and std for method Ours
# Average: 0.26967787969680057
# The average clip score and std for method SDS
# Average: 0.25994693816654263
# The average clip score and std for method VSD
# Average: 0.24547429160466275

# The average clip score and std for method Ours
# Average: 0.31781548394097225
# The average clip score and std for method SDS
# Average: 0.31178094773065484
# The average clip score and std for method VSD
# Average: 0.2996658567398313