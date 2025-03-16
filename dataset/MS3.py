import os
from os.path import join
# from wave import _wave_params
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

import numpy as np
import pandas as pd
import pickle
import json
import random
import cv2
from PIL import Image
from torchvision import transforms as T
from torchvision import transforms
from dataclasses import dataclass
import transformers
from typing import Sequence,Dict
from torch.nn.utils.rnn import pad_sequence



class MS3Dataset(Dataset):
    def __init__(
        self,
        split,
        # image_processor,
        num_frames=8,
        image_aspect_ratio='pad',
        data_root='/data/users/henghui_du/data/AVSBench-semantic',
        meta_csv_path='/data/users/henghui_du/Unified/data/metadata.csv',
    ):
        super(MS3Dataset, self).__init__()
        self.split = split
        self.data_root=data_root
        self.image_aspect_ratio=image_aspect_ratio
        # self.image_processor=image_processor
        self.num_frames=num_frames
        df_all = pd.read_csv(meta_csv_path, sep=',')
        df_all=df_all[df_all['label']=='v1m'] ## v1m
        self.df_split = df_all[df_all['split'] == split] ## split
        
        
    def __getitem__(self, index):

        df_one_video = self.df_split.iloc[index]
        video_name, set = df_one_video['uid'], df_one_video['label']

        sel_idx=random.randint(0,4)
        # image_path=join(self.data_root,set,video_name,'frames',f'{sel_idx}.jpg')
        # image=process_image(
        #     image_path=image_path,
        #     processor=self.image_processor,
        #     aspect_ratio=self.image_aspect_ratio,
        # )
        # # video=image.repeat(self.num_frames,1,1,1)
        # video=image
        
        # load iamge feature
        image_feature_path=join(self.data_root,'openai_clip-vit-large-patch14-336',video_name,f'{sel_idx}.npy')
        image_feat=np.load(image_feature_path)
        video = torch.tensor(image_feat,dtype=torch.float32) # 1,576,1024
        # video = image_feat.repeat(self.num_frames,1,1)

        audio_path=os.path.join(self.data_root,set,video_name,'audio_feat.npy')
        audio_feat=np.load(audio_path)
        audio_feat=audio_feat[sel_idx:sel_idx+1]
        audio_feat=torch.tensor(audio_feat,dtype=torch.float32)
        # audio_feat=audio_feat.repeat(self.num_frames,1)

        # label_dir=os.path.join(self.data_root, set, video_name, 'labels_semantic')
        # label_path=join(label_dir, "%d.png" % (sel_idx))
        # label=Image.open(label_path).convert('RGB')
        # label=expand2square(label,background_color=(0,0,0))
        # label=label.resize((336,336),Image.NEAREST)
        # # process
        # label=np.array(label)
        # label=label/127.5-1
        # label=torch.tensor(label,dtype=torch.float32).permute(2,0,1) # c,h,w

        target_token_path=join(self.data_root,'v1m_336_target_tokens',video_name,f'{sel_idx}.npy')
        target_tokens=np.load(target_token_path)
        target_tokens=torch.tensor(target_tokens,dtype=torch.long) # L

        # ori_image=Image.open(image_path).convert('RGB')
        # ori_image=expand2square(ori_image,background_color=(0,0,0))
        # ori_image=ori_image.resize((336,336),Image.BICUBIC)

        # instruction=f"[INST] Based on the video <video> and audio <audio>, please segment the area corresponding to the audio on the video frames. [/INST]"
        instruction=f"[INST] Based on the image <video> and audio <audio>, please segment the area corresponding to the audio on the image. The segmented area is represented by a pixel value of 255, and the other areas are represented by a pixel value of 0. [/INST]"
        # output=f'According to the video and audio, the segment area is <mask>.'
        output=f'According to the image and audio, the segment area is <mask>.'

        return {
            'instruction':instruction,
            'output':output,
            'video':video,
            'target_tokens':target_tokens,
            'audio':audio_feat,
            'video_name':video_name,
            'sel_idx':sel_idx,
            # 'ori_image':ori_image
        }
        

    def __len__(self):
        return len(self.df_split)

'''
full frames + enhance
1480 training samples
320 test samples
'''
class MS3Dataset_2(Dataset):
    def __init__(
        self,
        split,
        # image_processor,
        num_frames=8,
        image_aspect_ratio='pad',
        data_root='/data/users/henghui_du/data/AVSBench-semantic',
        meta_csv_path='/data/users/henghui_du/Unified/data/metadata.csv',
    ):
        super(MS3Dataset_2, self).__init__()
        self.split = split
        self.data_root=data_root
        self.image_aspect_ratio=image_aspect_ratio
        # self.image_processor=image_processor
        self.num_frames=num_frames
        df_all = pd.read_csv(meta_csv_path, sep=',')
        df_all=df_all[df_all['label']=='v1m'] ## v1m
        df_split = df_all[df_all['split'] == split] ## split
        
        vnums=len(df_split)
        data=[]
        for i in range(vnums):
            df_one_video = df_split.iloc[i]
            video_name, set = df_one_video['uid'], df_one_video['label']
            for sel_idx in range(5):
                image_feature_path=join(self.data_root,'openai_clip-vit-large-patch14-336',video_name,f'{sel_idx}.npy')
                audio_path=os.path.join(self.data_root,set,video_name,'audio_feat.npy')
                target_token_path=join(self.data_root,'v1m_336_target_tokens_enhance',video_name,f'{sel_idx}.npy')

                data.append(
                    {
                        'image_feature_path':image_feature_path,
                        'audio_path':audio_path,
                        'target_token_path':target_token_path,
                        'video_name':video_name,
                        'sel_idx':sel_idx
                    }
                )
        self.data=data


    def __getitem__(self, index):

        sample=self.data[index]
        image_feature_path=sample['image_feature_path']
        video_name=sample['video_name']
        sel_idx=sample['sel_idx']

        # image_path=join(self.data_root,'v1m',video_name,'frames',f'{sel_idx}.jpg')
        # ori_image=Image.open(image_path).convert('RGB')
        # ori_image=expand2square(ori_image,(0,0,0))
        # ori_image = ori_image.resize((336,336),Image.NEAREST)
        # ori_image=ori_image.convert('L')
        # ori_image=np.array(ori_image)
        # ori_image = ori_image>0
        # gt_binary_mask = torch.as_tensor(ori_image>0,dtype=torch.float32)
        # gt_binary_mask = gt_binary_mask.unsqueeze(0) # 1,336,336
        
        # load iamge feature
        image_feat=np.load(image_feature_path)
        video = torch.tensor(image_feat,dtype=torch.float32) # 1,576,1024

        audio_path=sample['audio_path']
        audio_feat=np.load(audio_path)
        audio_feat=audio_feat[sel_idx:sel_idx+1]
        audio_feat=torch.tensor(audio_feat,dtype=torch.float32)

        target_token_path=sample['target_token_path']
        target_tokens=np.load(target_token_path)
        target_tokens=torch.tensor(target_tokens,dtype=torch.long) # L

        # instruction=f"[INST] Based on the video <video> and audio <audio>, please segment the area corresponding to the audio on the video frames. [/INST]"
        instruction=f"[INST] Based on the image <video> and audio <audio>, please segment the area corresponding to the audio in the image. [/INST]"
        output=f'According to the video and audio, the segment area is <mask>.'
        # output=f'According to the image and audio, the segment area is <mask>.'
        # instruction=f'[INST] Based on the image <video> and audio <audio>, please segment the area corresponding to the audio in the image. [/INST]'
        # output=f'According to the image and audio, the segment area is <seg>.'

        return {
            'instruction':instruction,
            'output':output,
            'video':video,
            'target_tokens':target_tokens,
            'audio':audio_feat,
            'video_name':video_name,
            'sel_idx':sel_idx,
            # 'ori_image':gt_binary_mask
        }
        

    def __len__(self):
        return len(self.data)



@dataclass
class DataCollatorForMS3Dataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        
        tokenizer=self.tokenizer

        batch_input_ids=[]
        batch_label=[]
        batch_mask=[]
        
        batch_video=[]
        batch_audio=[]
        batch_target_tokens=[]
        batch_video_name=[]
        batch_sel_idx=[]
        # batch_ori_image=[]

        for instance in instances:

            instruction=instance['instruction']
            output=instance['output']
            video=instance['video']
            audio=instance['audio']
            target_tokens=instance['target_tokens']
            video_name=instance['video_name']
            sel_idx=instance['sel_idx']
            # ori_image=instance['ori_image']

            instruction_ids = [tokenizer.bos_token_id]+tokenizer.convert_tokens_to_ids(tokenizer.tokenize(instruction))
            output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))+[tokenizer.eos_token_id]
            
            input_ids=instruction_ids+output_ids
            label=[-100]*len(instruction_ids)+output_ids
            mask=[1]*len(input_ids)
        
            batch_input_ids.append(torch.tensor(input_ids,dtype=torch.long))
            batch_label.append(torch.tensor(label,dtype=torch.long))
            batch_mask.append(torch.tensor(mask,dtype=torch.int32))
            
            batch_audio.append(audio)
            batch_video.append(video)
            batch_target_tokens.append(target_tokens)
            batch_video_name.append(video_name)
            batch_sel_idx.append(sel_idx)
            # batch_ori_image.append(ori_image)

        batch_input_ids = pad_sequence(batch_input_ids,batch_first=True,padding_value=tokenizer.pad_token_id)
        batch_label = pad_sequence(batch_label,batch_first=True,padding_value=-100)
        batch_mask = pad_sequence(batch_mask,batch_first=True,padding_value=0)
        batch_video = torch.stack(batch_video,dim=0) # b,t,c,h,w  # b,t,N,D
        batch_audio = torch.stack(batch_audio,dim=0) # b,t,d
        batch_target_tokens=torch.stack(batch_target_tokens,dim=0) # b,L
        # batch_ori_image=torch.stack(batch_ori_image,dim=0)
        
        return {
            'input_ids':batch_input_ids,
            'labels':batch_label,
            'attention_mask':batch_mask,
            'video':batch_video,
            'audio':batch_audio,
            'target_tokens':batch_target_tokens,
            'batch_video_name':batch_video_name,
            'batch_sel_idx':batch_sel_idx,
            # 'batch_ori_image':batch_ori_image,
        }



@dataclass
class DataCollatorForMS3TestDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        
        tokenizer=self.tokenizer

        batch_input_ids=[]
        batch_label=[]
        batch_mask=[]
        
        batch_video=[]
        batch_audio=[]
        # batch_target_tokens=[]
        batch_video_name=[]
        batch_sel_idx=[]
        batch_ori_image=[]

        for instance in instances:

            instruction=instance['instruction']
            output=instance['output']
            video=instance['video']
            audio=instance['audio']
            # target_tokens=instance['target_tokens']
            video_name=instance['video_name']
            sel_idx=instance['sel_idx']
            ori_image=instance['ori_image']

            instruction_ids = [tokenizer.bos_token_id]+tokenizer.convert_tokens_to_ids(tokenizer.tokenize(instruction))
            output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))+[tokenizer.eos_token_id]
            
            input_ids=instruction_ids
            label=[-100]*len(instruction_ids)
            mask=[1]*len(input_ids)
        
            batch_input_ids.append(torch.tensor(input_ids,dtype=torch.long))
            batch_label.append(torch.tensor(label,dtype=torch.long))
            batch_mask.append(torch.tensor(mask,dtype=torch.int32))
            
            batch_audio.append(audio)
            batch_video.append(video)
            # batch_target_tokens.append(target_tokens)
            batch_video_name.append(video_name)
            batch_sel_idx.append(sel_idx)
            batch_ori_image.append(ori_image)

        batch_input_ids = pad_sequence(batch_input_ids,batch_first=True,padding_value=tokenizer.pad_token_id)
        batch_label = pad_sequence(batch_label,batch_first=True,padding_value=-100)
        batch_mask = pad_sequence(batch_mask,batch_first=True,padding_value=0)
        batch_video = torch.stack(batch_video,dim=0) # b,t,c,h,w  # b,t,N,D
        batch_audio = torch.stack(batch_audio,dim=0) # b,t,d
        # batch_target_tokens=torch.stack(batch_target_tokens,dim=0) # b,L
        batch_ori_image=torch.stack(batch_ori_image,dim=0)
        
        return {
            'input_ids':batch_input_ids,
            'labels':batch_label,
            'attention_mask':batch_mask,
            'video':batch_video,
            'audio':batch_audio,
            # 'target_tokens':batch_target_tokens,
            'batch_video_name':batch_video_name,
            'batch_sel_idx':batch_sel_idx,
            'batch_ori_image':batch_ori_image,
        }



def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = MS3Dataset(
        split='train',
        # image_processor=data_args.image_processor,
        num_frames=data_args.num_frames,
        image_aspect_ratio=data_args.image_aspect_ratio,
    )
    data_collator = DataCollatorForMS3Dataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)



def make_supervised_data_module_2(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = MS3Dataset_2(
        split='train',
        # image_processor=data_args.image_processor,
        num_frames=data_args.num_frames,
        image_aspect_ratio=data_args.image_aspect_ratio,
    )
    data_collator = DataCollatorForMS3Dataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)




