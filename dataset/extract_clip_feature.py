import os,sys
sys.path.append(os.getcwd())
from tqdm import tqdm
from os.path import join
import torch
import numpy as np

from models.clip_encoder import CLIPVisionTower
from utils.mm_utils import process_image

model=CLIPVisionTower()
model.eval()
model.cuda()

data_root='/data/users/henghui_du/data/AVSBench-semantic'
set='v1s'
vnames=os.listdir(join(data_root,set))
pabr=tqdm(total=len(vnames),desc='Extract CLIP Feature')

for vname in vnames:
    tot=10
    if set=='v1m':
        tot=5
    elif set=='v1s':
        tot=1
    for i in range(tot):
        image_path=join(data_root,set,vname,'frames',f'{i}.jpg')
        if not os.path.exists(image_path):
            continue
        image=process_image(
            image_path=image_path,
            processor=model.image_processor,
            aspect_ratio='none',  # no padding
        )

        with torch.no_grad():
            feature=model(image)
        
        feature=feature.cpu().data.numpy()
        save_dir=join(data_root,f'{set}_openai_clip-vit-large-patch14-336_no_padding',vname)
        os.makedirs(save_dir,exist_ok=True)
        np.save(join(save_dir,f'{i}.npy'),feature) # 1,576,1024
    
    pabr.update(1)


