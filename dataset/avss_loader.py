import os
# from wave import _wave_params
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

import numpy as np
import pandas as pd
import pickle
import json
import random
# import cv2
from PIL import Image
from torchvision import transforms

# from .config import cfg_avs


def get_v2_pallete(label_to_idx_path, num_cls=71):
    def _getpallete(num_cls=71):
        """build the unified color pallete for AVSBench-object (V1) and AVSBench-semantic (V2),
        71 is the total category number of V2 dataset, you should not change that"""
        n = num_cls
        pallete = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            pallete[j * 3 + 0] = 0
            pallete[j * 3 + 1] = 0
            pallete[j * 3 + 2] = 0
            i = 0
            while (lab > 0):
                pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i = i + 1
                lab >>= 3
        return pallete  # list, lenth is n_classes*3

    with open(label_to_idx_path, 'r') as fr:
        label_to_pallete_idx = json.load(fr)
    v2_pallete = _getpallete(num_cls)  # list
    v2_pallete = np.array(v2_pallete).reshape(-1, 3)
    assert len(v2_pallete) == len(label_to_pallete_idx)
    return v2_pallete


def crop_resize_img(crop_size, img, img_is_mask=False):
    outsize = crop_size
    short_size = outsize
    w, h = img.size
    if w > h:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    else:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    if not img_is_mask:
        img = img.resize((ow, oh), Image.BILINEAR)
    else:
        img = img.resize((ow, oh), Image.NEAREST)
    # center crop
    w, h = img.size
    x1 = int(round((w - outsize) / 2.))
    y1 = int(round((h - outsize) / 2.))
    img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
    # print("crop for train. set")
    return img


def resize_img(crop_size, img, img_is_mask=False):
    outsize = crop_size
    # only resize for val./test. set
    if not img_is_mask:
        img = img.resize((outsize, outsize), Image.BILINEAR)
    else:
        img = img.resize((outsize, outsize), Image.NEAREST)
    return img


def color_mask_to_label(mask, v_pallete):
    mask_array = np.array(mask).astype('int32')
    semantic_map = []
    for colour in v_pallete:
        equality = np.equal(mask_array, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    # pdb.set_trace() # there is only one '1' value for each pixel, run np.sum(semantic_map, axis=-1)
    label = np.argmax(semantic_map, axis=-1)
    return label


def load_image_in_PIL_to_Tensor(path, split='train', mode='RGB', transform=None, cfg=None):
    img_PIL = Image.open(path).convert(mode)
    if cfg.crop_img_and_mask:
        if split == 'train':
            img_PIL = crop_resize_img(
                cfg.crop_size, img_PIL, img_is_mask=False)
        else:
            img_PIL = resize_img(cfg.crop_size,
                                 img_PIL, img_is_mask=False)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL


def load_color_mask_in_PIL_to_Tensor(path, v_pallete, split='train', mode='RGB', cfg=None):
    color_mask_PIL = Image.open(path).convert(mode)
    if cfg.crop_img_and_mask:
        if split == 'train':
            color_mask_PIL = crop_resize_img(
                cfg.crop_size, color_mask_PIL, img_is_mask=True)
        else:
            color_mask_PIL = resize_img(
                cfg.crop_size, color_mask_PIL, img_is_mask=True)
    # obtain semantic label
    color_label = color_mask_to_label(color_mask_PIL, v_pallete)
    color_label = torch.from_numpy(color_label)  # [H, W]
    color_label = color_label.unsqueeze(0)
    # binary_mask = (color_label != (cfg_avs.NUM_CLASSES-1)).float()
    # return color_label, binary_mask # both [1, H, W]
    return color_label  # both [1, H, W]


def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach()  # [5, 1, 96, 64]
    return audio_log_mel



dir_base='/data/users/henghui_du/data/AVSBench-semantic'
meta_csv_path='/data/users/henghui_du/Unified/data/metadata.csv'
label_idx_path='/data/users/henghui_du/data/AVSBench-semantic/label2idx.json'
mask_num=10
num_classes=71
size=(256,256)

class AVS_V2_Dataset(Dataset):
    def __init__(self, split, prompt_feat_path):
        super(AVS_V2_Dataset, self).__init__()
        self.split = split
        self.mask_num = mask_num
        df_all = pd.read_csv(meta_csv_path, sep=',')
        df_all=df_all[df_all['label']=='v2'] ## avss
        self.df_split = df_all[df_all['split'] == split] ## split
        print("{}/{} videos are used for {}.".format(len(self.df_split),
              len(df_all), self.split))
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.v2_pallete = get_v2_pallete(label_idx_path, num_cls=num_classes)
        self.label_transform=transforms.Resize((224,224),transforms.InterpolationMode.NEAREST)
        
        self.prompt_feat=torch.tensor(np.load(prompt_feat_path)) # 1,768


    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name, set = df_one_video['uid'], df_one_video['label']

        level_0_features=[]
        level_1_features=[]
        level_2_features=[]
        level_3_features=[]
        if self.split=='train':
            sel_idx=random.randint(0,9)
            level_0_path=os.path.join(dir_base,'v2_pvt_224x224_feature',video_name,f'frame_{sel_idx}_feature_level_0.npy')
            level_1_path=os.path.join(dir_base,'v2_pvt_224x224_feature',video_name,f'frame_{sel_idx}_feature_level_1.npy')
            level_2_path=os.path.join(dir_base,'v2_pvt_224x224_feature',video_name,f'frame_{sel_idx}_feature_level_2.npy')
            level_3_path=os.path.join(dir_base,'v2_pvt_224x224_feature',video_name,f'frame_{sel_idx}_feature_level_3.npy')
            level_0_features.append(torch.tensor(np.load(level_0_path)))
            level_1_features.append(torch.tensor(np.load(level_1_path)))
            level_2_features.append(torch.tensor(np.load(level_2_path)))
            level_3_features.append(torch.tensor(np.load(level_3_path)))

            path=os.path.join('/data/users/henghui_du/data/AVSBench-semantic/v2_vit_b32_feature',video_name+'.npy')
            data=np.load(path)
            visual_feat=data[sel_idx:sel_idx+1]
            
            # for i in range(10):
            #     level_0_path=os.path.join(dir_base,'v2_pvt_224x224_feature',video_name,f'frame_{i}_feature_level_0.npy')
            #     level_3_path=os.path.join(dir_base,'v2_pvt_224x224_feature',video_name,f'frame_{i}_feature_level_3.npy')
            #     level_0_features.append(torch.tensor(np.load(level_0_path)))
            #     level_3_features.append(torch.tensor(np.load(level_3_path)))

            # path=os.path.join('/data/users/henghui_du/data/AVSBench-semantic/v2_vit_b32_feature',video_name+'.npy')
            # visual_feat=np.load(path)
        else:
            for i in range(10):
                sel_idx=i
                level_0_path=os.path.join(dir_base,'v2_pvt_224x224_feature',video_name,f'frame_{sel_idx}_feature_level_0.npy')
                level_1_path=os.path.join(dir_base,'v2_pvt_224x224_feature',video_name,f'frame_{sel_idx}_feature_level_1.npy')
                level_2_path=os.path.join(dir_base,'v2_pvt_224x224_feature',video_name,f'frame_{sel_idx}_feature_level_2.npy')
                level_3_path=os.path.join(dir_base,'v2_pvt_224x224_feature',video_name,f'frame_{sel_idx}_feature_level_3.npy')
                level_0_features.append(torch.tensor(np.load(level_0_path)))
                level_1_features.append(torch.tensor(np.load(level_1_path)))
                level_2_features.append(torch.tensor(np.load(level_2_path)))
                level_3_features.append(torch.tensor(np.load(level_3_path)))

            path=os.path.join('/data/users/henghui_du/data/AVSBench-semantic/v2_vit_b32_feature',video_name+'.npy')
            visual_feat=np.load(path)

        level_0_features=torch.stack(level_0_features,dim=0)
        level_1_features=torch.stack(level_1_features,dim=0)
        level_2_features=torch.stack(level_2_features,dim=0)
        level_3_features=torch.stack(level_3_features,dim=0)
        visual_feat=torch.tensor(visual_feat) # b,t,512

        audio_path=os.path.join(dir_base,'v2',video_name,'audio_feat.npy')
        audio_feat=np.load(audio_path)
        if self.split=='train':
            audio_feat=audio_feat[sel_idx:sel_idx+1]
        audio_feat=torch.tensor(audio_feat)

        color_mask_base_path = os.path.join(dir_base, set, video_name, 'labels_rgb')

        if self.split=='train':
            gt_temporal_mask_flag = torch.ones(1)  # .bool()
        else:
            gt_temporal_mask_flag = torch.ones(10)

        labels = []
        if self.split=='train':
            mask_path = os.path.join(color_mask_base_path, "%d.png" % (sel_idx))
            mask=Image.open(mask_path).convert('RGB')
            mask=mask.resize((256,256),Image.NEAREST)
            # mask=color_mask_to_label(mask,self.v2_pallete)
            mask=np.array(mask)
            # mask=mask/255.
            mask=torch.from_numpy(mask)
            mask=mask.unsqueeze(0) # 1,h,w
            labels.append(mask)

            # for i in range(10):
            #     mask_path = os.path.join(color_mask_base_path, "%d.png" % (i))
            #     mask=Image.open(mask_path).convert('RGB')
            #     mask=mask.resize(size,Image.NEAREST)
            #     mask=color_mask_to_label(mask,self.v2_pallete)
            #     mask=torch.from_numpy(mask)
            #     mask=mask.unsqueeze(0) # 1,h,w
            #     labels.append(mask)
        else:
            for mask_id in range(10):
                mask_path = os.path.join(color_mask_base_path, "%d.png" % (mask_id))
                mask=Image.open(mask_path).convert('RGB')
                mask=mask.resize(size,Image.NEAREST)
                mask=color_mask_to_label(mask,self.v2_pallete)
                mask=torch.from_numpy(mask)
                mask=mask.unsqueeze(0) # 1,h,w
                labels.append(mask)

        labels_tensor = torch.stack(labels, dim=0)

        if self.split=='train':
            prompt_feat=self.prompt_feat
        else:
            prompt_feat=self.prompt_feat

        return {
            'level_0_features':level_0_features, # t,c,h,w
            'level_3_features':level_3_features,
            'level_1_features':level_1_features, # t,c,h,w
            'level_2_features':level_2_features,
            'labels':labels_tensor, # t,1,224,224
            'audio_feat':audio_feat, # t,128
            'visual_feat':visual_feat, # t,512
            'prompt_feat':prompt_feat, # 1,512
            'task_name':'avss',
            'gt_temporal_mask_flag':gt_temporal_mask_flag,
            'vname':video_name,
            'sel_idx':sel_idx
        }
        

    def __len__(self):
        return len(self.df_split)

   

def get_loaders(args):
    train_dataset=AVS_V2_Dataset(split='train',prompt_feat_path=args.avs_prompt_feat_path)
    # val_dataset=AVS_V2_Dataset(split='val',prompt_feat_path=args.avs_prompt_feat_path)
    test_dataset=AVS_V2_Dataset(split='test',prompt_feat_path=args.avs_prompt_feat_path)
    
    train_loader=DataLoader(
        dataset=train_dataset,
        batch_size=args.avss_train_bs,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    # val_loader=DataLoader(
    #     dataset=val_dataset,
    #     batch_size=args.avss_val_bs,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True,
    #     drop_last=False
    # )
    test_loader=DataLoader(
        dataset=test_dataset,
        batch_size=args.avss_test_bs,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    return train_loader,test_loader

