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

from cleanfid import fid
# score = fid.compute_fid('results/eval/05-19/vsd_gen__/lr0.010_seed0_scale7.5', 'results/eval/real')
score = fid.compute_fid('results/eval/05-19/vsd_gen__/lr0.010_seed0_scale7.5', 'results/eval/real', mode="clean", model_name="clip_vit_b_32")
# print('vsd fid', score) # vsd fid 65.76529093923341
print('vsd clip fid', score) # vsd clip fid 20.50432097705861
# score = fid.compute_fid('results/eval/05-18/sds_gen__/lr0.100_seed0_scale100.0', 'results/eval/real')
score = fid.compute_fid('results/eval/05-18/sds_gen__/lr0.100_seed0_scale100.0', 'results/eval/real', mode="clean", model_name="clip_vit_b_32")
# print('sds fid', score) # sds fid 79.95757698810667
print('sds clip fid', score) # sds clip fid 27.90741242730921
# score = fid.compute_fid('results/eval/05-18/nfsd_gen__/lr0.100_seed0_scale7.5', 'results/eval/real')
score = fid.compute_fid('results/eval/05-18/nfsd_gen__/lr0.100_seed0_scale7.5', 'results/eval/real', mode="clean", model_name="clip_vit_b_32")
# print('nfsd fid', score) # nfsd fid 79.68118533402685
print('nfsd clip fid', score) # nfsd clip fid 27.579900820536338
# score = fid.compute_fid('results/eval/05-18/sds++_gen__/lr0.100_seed0_scale100.0', 'results/eval/real')
score = fid.compute_fid('results/eval/05-18/sds++_gen__/lr0.100_seed0_scale100.0', 'results/eval/real', mode="clean", model_name="clip_vit_b_32")
# print('sds fid', score) # sds fid 77.25809794821657
print('sds clip fid', score) # sds clip fid 27.630883496976082
# score = fid.compute_fid('results/eval/05-21/sd_sds_gen_step/a_DSLR_photo_of_a_dog_in_a_winter_wonderland_lr0.010_seed48/', 'results/eval/real')
score = fid.compute_fid('results/eval/05-21/sd_sds_gen_step/a_DSLR_photo_of_a_dog_in_a_winter_wonderland_lr0.010_seed48/', 'results/eval/real', mode="clean", model_name="clip_vit_b_32")
# print('ours fid', score) # ours fid 69.82971962872455
print('ours clip fid', score) # ours clip fid 21.56211584094494

score = fid.compute_fid('results/eval/05-21/ddim_gen__/lr0.010_seed0_scale100.0_zero', 'results/eval/real')
print('ddim fid', score) # ddim fid 47.596365331
score = fid.compute_fid('results/eval/05-21/ddim_gen__/lr0.010_seed0_scale100.0_zero', 'results/eval/real', mode="clean", model_name="clip_vit_b_32")
print('ddim clip fid', score) # ddim clip fid 16.518581578872684

score = fid.compute_fid('results/eval/05-22/sd_csd_gen_step/a_DSLR_photo_of_a_dog_in_a_winter_wonderland_lr0.010_seed48/', 'results/eval/real')
score = fid.compute_fid('results/eval/05-22/sd_sds_gen_step/a_DSLR_photo_of_a_dog_in_a_winter_wonderland_lr0.010_seed48/1500/', 'results/eval/real')
print('csd fid', score) # csd fid 47.596365331
score = fid.compute_fid('results/eval/05-22/sd_csd_gen_step/a_DSLR_photo_of_a_dog_in_a_winter_wonderland_lr0.010_seed48/', 'results/eval/real', mode="clean", model_name="clip_vit_b_32")
print('csd clip fid', score) # csd clip fid 16.518581578872684

score = fid.compute_fid('results/eval/05-28/ddim_gen__/lr0.010_seed0_scale100.0_zero', 'results/eval/real_val')
score = fid.compute_fid('results/eval/05-28/ddim_gen__/lr0.010_seed0_scale100.0_zero', 'results/eval/real_val', mode="clean", model_name="clip_vit_b_32")
