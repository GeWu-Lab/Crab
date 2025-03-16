
import os,sys
from os.path import join
sys.path.append(os.getcwd())
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from dataset.MS3 import MS3Dataset
from torchvision import transforms

from utils.mm_utils import expand2square
from models.taming_transformer.vqgan import VQModel

# image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336',local_files_only=True)
# vision_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14-336')

ddconfig={
    'double_z':False,
    'z_channels':256,
    'resolution':256,
    'in_channels':3,
    'out_ch':3,
    'ch':128,
    'ch_mult':(1,1,2,2,4),
    'num_res_blocks':2,
    'attn_resolutions':(16,),
    'dropout':0.0
}

vqgan=VQModel(
    ddconfig=ddconfig,
    lossconfig=None,
    n_embed=16384,
    embed_dim=256,
    ckpt_path='/data/users/henghui_du/pretrain/vqgan_imagenet_f16_16384/ckpts/last.ckpt'
)
vqgan.eval()
vqgan.cuda()

data_root='/data/users/henghui_du/data/AVSBench-semantic'
set='v1m'
vnames=os.listdir(join(data_root,set))
pbar=tqdm(total=len(vnames),desc='Extract target tokens')

for vname in vnames:
    save_dir=join(data_root,f'{set}_336_image_vqgan_tokens_padding',vname)
    tot=10
    if set=='v1m':
        tot=5
    elif set=='v1s':
        tot=5
    for i in range(tot):
        # if os.path.exists(join(save_dir,f'{i}.npy')):
        #     continue
        path=join(data_root,set,vname,'frames',f'{i}.jpg')
        label=Image.open(path).convert('RGB')
        label=expand2square(label,background_color=(0,0,0))  # padding
        label=label.resize((336,336),Image.BICUBIC)
        label=torch.tensor(np.array(label))
        # label.masked_fill_(label>0.,255)
        label=label/127.5-1.
        label=torch.tensor(label,dtype=torch.float32).permute(2,0,1).contiguous()

    
        with torch.no_grad():
            # dec,_=vqgan(label)
            indices=vqgan.get_codebook_indices(label.unsqueeze(0).cuda())
        # print(indices.shape) # 1,441
        
        indices=indices[0].cpu().data.numpy()
        os.makedirs(save_dir,exist_ok=True)
        np.save(join(save_dir,f'{i}.npy'),indices)


    pbar.update(1)
    

