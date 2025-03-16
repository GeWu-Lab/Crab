import json
import ast
import os
from os.path import join,exists
import numpy as np
import pandas as pd
import cv2,csv
from typing import Sequence,Dict
from dataclasses import dataclass
import librosa
from PIL import Image
import torch
import random
import transformers
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from decord import VideoReader
from transformers import CLIPImageProcessor

from dataset.audio_processor import preprocess


'''
AVQA dataset:
    - train: 31927
    - test: 9129
MS3 dataset:
    - train: 1480
    - test: 320
'''

label_to_idx_path = '/group/40061/cserdu/data/AVS/label2idx.json'

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


class UnifiedTestDataset(Dataset):
    def __init__(
        self,
        mode='test', # train,val,test
        video_processor: CLIPImageProcessor = None,
        tokenizer: PreTrainedTokenizer = None,
        image_size = 224,
        video_frame_nums = 10,
        # avqa
        avqa_task=False,
        # ave task
        ave_task = False,
        # avvp task
        avvp_task = False,
        # avs task
        avss_task = False,
        ms3_task=False,
        s4_task=False,
        multi_frames = False,
        image_scale_nums = 2,
        token_nums_per_scale = 3,
        ref_avs_task = False,
        test_name = 'test_s',  # for ref-avs: test_s, test_u, test_n
        # audio referred image grounding task
        arig_task = False,
    ) -> None:
        super().__init__()

        self.mode=mode
        self.video_processor = video_processor
        self.multi_frames = multi_frames
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.video_frame_nums = video_frame_nums
        self.test_name = test_name

        # if avss_task or ms3_task or s4_task or ref_avs_task:
        token_nums = image_scale_nums * token_nums_per_scale
        mask_token = [f'<mask_{i}>' for i in range(token_nums)]
        self.mask_token = ''.join(mask_token)
        print('mask token: ',self.mask_token)

        self.samples = []
        self.tot = 0

        if avqa_task:
            self.test_task = 'avqa'
        elif ave_task:
            self.test_task = 'ave'
        elif avvp_task:
            self.test_task = 'avvp'
        elif arig_task:
            self.test_task = 'arig'
        elif s4_task:
            self.test_task = 's4'
        elif ms3_task:
            self.test_task = 'ms3'
        elif avss_task:
            self.test_task = 'avss'
        elif ref_avs_task:
            self.test_task = 'ref-avs'
        else:
            raise ValueError('invalid task.')
        
        print('test_task: ',self.test_task)

        self.add_custome_test_samples()

        if avss_task:
            self.v2_pallete = get_v2_pallete(label_to_idx_path=label_to_idx_path,num_cls=71)


    def add_custome_test_samples(self):
        with open('data/example.json','r') as f:
            samples = json.load(f)
        for sample in samples:
            task = sample['task']
            if task != self.test_task:
                continue
            if task == 'avqa':
                audio_path = sample['audio_path']
                video_path = sample['video_path']
                question = sample['question']
                instruction = f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease answer this question: {question}'
                self.samples.append(
                    {
                        'video_path':video_path,
                        'audio_path':audio_path,
                        'instruction':instruction,
                        'output':'none',
                        'task_name':'avqa'
                    }
                )
            elif task == 'ave':
                audio_path = sample['audio_path']
                video_path = sample['video_path']
                instruction = f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease describe the events and time range that occurred in the video.'
                self.samples.append({
                    'audio_path':audio_path,
                    'video_path':video_path,
                    'instruction':instruction,
                    'output':'none',
                    'task_name':'ave'
                })
            elif task == 'avvp':
                audio_path = sample['audio_path']
                video_path = sample['video_path']
                instruction = f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease determine the events that occur based on the visual and audio information, as well as the start and end time of these events.'
                self.samples.append({
                    'audio_path':audio_path,
                    'video_path':video_path,
                    'instruction':instruction,
                    'output':'none',
                    'task_name':'avvp'
                })

            elif task == 'arig':
                audio_path = sample['audio_path']
                image_path = sample['image_path']
                idx = int(image_path.split('/')[-1][:-4])
                tot = 5
                instruction = f'This is an image:\n<image_start><image><image_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease output the location coordinates of sounding object.'
                self.samples.append({
                    'audio_path':audio_path,
                    'image_path':image_path,
                    'instruction':instruction,
                    'idx':idx,
                    'tot':tot,
                    'output':'none',
                    'task_name':'arig'
                })

            elif task == 's4':
                audio_path = sample['audio_path']
                image_path = sample['image_path']
                mask_path = sample['mask_path']
                instruction = f'This is an image:\n<image_start><image><image_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease segment out the object that makes the sound in the image.'
                idx = int(image_path.split('/')[-1][:-4])
                tot = 5
                self.samples.append({
                    'audio_path':audio_path,
                    'image_path':image_path,
                    'instruction':instruction,
                    'mask_path': mask_path,
                    'output':'none',
                    'task_name':'s4',
                    'idx':idx,
                    'tot':tot
                })
            elif task == 'ms3':
                audio_path = sample['audio_path']
                image_path = sample['image_path']
                mask_path = sample['mask_path']
                instruction = f'This is an image:\n<image_start><image><image_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease segment out the object that makes the sound in the image.'
                idx = int(image_path.split('/')[-1][:-4])
                tot = 5
                self.samples.append({
                    'audio_path':audio_path,
                    'image_path':image_path,
                    'instruction':instruction,
                    'mask_path': mask_path,
                    'output':'none',
                    'task_name':'ms3',
                    'idx':idx,
                    'tot':tot
                })
            elif task == 'avss':
                audio_path = sample['audio_path']
                image_path = sample['image_path']
                mask_path = sample['mask_path']
                idx = int(image_path.split('/')[-1][:-4])
                instruction = f'This is an image:\n<image_start><image><image_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease segment out the object that makes the sound in the image.'
                self.samples.append({
                    'image_path':image_path,
                    'audio_path':audio_path,
                    'mask_path':mask_path,
                    'instruction':instruction,
                    'output':'none',
                    'idx':idx,
                    'task_name':'avss'
                })
            elif task == 'ref-avs':
                audio_path = sample['audio_path']
                image_path = sample['image_path']
                mask_path = sample['mask_path']
                exp = sample['exp']
                instruction = f'This is an image:\n<image_start><image><image_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease segment out {exp.lower()} in the image.'
                self.samples.append({
                    'image_path':image_path,
                    'audio_path':audio_path,
                    'mask_path':mask_path,
                    'instruction': instruction,
                    'output': 'none',
                    'task_name':'ref-avs'
                })

    def __len__(self):
        return len(self.samples)


    def __getitem__(self,idx):
        sample = self.samples[idx]
        task_name = sample['task_name']
        instruction = sample['instruction']
        output = sample.get('output',None)
        if output is None:
            label_path = sample['label_path']
            output = self.read_label(label_path)
        if self.tokenizer is not None and hasattr(self.tokenizer,'apply_chat_template'):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction},
            ]
            instruction = self.tokenizer.apply_chat_template(conversation=messages,add_generation_prompt=True,tokenize=False)
            output = output + '</s>'
        
        data = {
            'instruction': instruction,
            'output': output,
            'task_name':task_name,
        }
        
        if task_name=='avqa':
            audio_path = sample['audio_path']
            video_path = sample['video_path']
            ### process video
            vr = VideoReader(uri=video_path, height=self.image_size, width=self.image_size)
            vlen = len(vr)
            start, end = 0, vlen
            n_frms = self.video_frame_nums
            n_frms = min(n_frms, vlen)
            indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
            # get_batch -> T, H, W, C
            temp_frms = vr.get_batch(indices).asnumpy()
            frames = []
            T = temp_frms.shape[0]
            for i in range(T):
                frame = Image.fromarray(temp_frms[i])
                frames.append(frame)
            frames = self.video_processor.preprocess(frames,return_tensors='pt')
            video = frames['pixel_values']  # t,c,h,w
            data['video'] = video
            data['video_path'] = video_path
            
            ### process audio
            audio_feature = []
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)
            length = len(audio)
            tot = 60
            nums_per_second = int(length / tot)
            indices = [i for i in range(0,60,6)]
            for indice in indices:
                start_time = max(0, indice - 0.5)
                end_time = min(tot, indice + 1.5)
                audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]
                if indice - 0.5 < 0:
                    sil = np.zeros(2 * nums_per_second - len(audio_seg), dtype=float)
                    audio_seg = np.concatenate((sil, audio_seg),axis=0)
                if indice + 1.5 > tot:
                    sil = np.zeros(2 * nums_per_second - len(audio_seg), dtype=float)
                    audio_seg = np.concatenate((audio_seg, sil),axis=0)
                audio_seg = torch.from_numpy(audio_seg).unsqueeze(0)
                fbank = preprocess(audio_seg)
                fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
                audio_feature.append(fbank)
            audio_feature = torch.stack(audio_feature,dim=0) # t,L,128
            data['audio'] = audio_feature
            data['audio_path'] = audio_path

        elif task_name == 'ave':
            audio_path = sample['audio_path']
            video_path = sample['video_path']
            ### process video
            vr = VideoReader(uri=video_path, height=self.image_size, width=self.image_size)
            vlen = len(vr)
            start, end = 0, vlen
            n_frms = self.video_frame_nums
            n_frms = min(n_frms, vlen)
            indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
            # get_batch -> T, H, W, C
            temp_frms = vr.get_batch(indices).asnumpy()
            frames = []
            T = temp_frms.shape[0]
            for i in range(T):
                frame = Image.fromarray(temp_frms[i])
                frames.append(frame)
            frames = self.video_processor.preprocess(frames,return_tensors='pt')
            video = frames['pixel_values']  # t,c,h,w
            data['video'] = video
            data['video_path'] = video_path
            
            ### process audio
            audio_feature = []
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)
            length = len(audio)
            tot = 10
            indices = [i for i in range(tot)]
            nums_per_second = int(length / tot)
            for indice in indices:
                start_time = max(0, indice)
                end_time = min(tot, indice + 1)
                audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]
                if len(audio_seg) < 1 * nums_per_second:
                    sil = np.zeros(1 * nums_per_second - len(audio_seg), dtype=float)
                    audio_seg = np.concatenate((audio_seg, sil),axis=0)
                audio_seg = torch.from_numpy(audio_seg).unsqueeze(0)
                fbank = preprocess(audio_seg)
                fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
                audio_feature.append(fbank)
            audio_feature = torch.stack(audio_feature,dim=0) # t,L,128
            data['audio'] = audio_feature
            data['audio_path'] = audio_path

        elif task_name == 'avvp':
            audio_path = sample['audio_path']
            video_path = sample['video_path']
            ### process video
            vr = VideoReader(uri=video_path, height=self.image_size, width=self.image_size)
            vlen = len(vr)
            start, end = 0, vlen
            n_frms = self.video_frame_nums
            n_frms = min(n_frms, vlen)
            indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
            # get_batch -> T, H, W, C
            temp_frms = vr.get_batch(indices).asnumpy()
            frames = []
            T = temp_frms.shape[0]
            for i in range(T):
                frame = Image.fromarray(temp_frms[i])
                frames.append(frame)
            frames = self.video_processor.preprocess(frames,return_tensors='pt')
            video = frames['pixel_values']  # t,c,h,w
            # data['video'] = video
            
            ### process audio
            audio_feature = []
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)
            length = len(audio)
            tot = 10
            nums_per_second = int(length / tot)
            indices = [i for i in range(10)]
            for indice in indices:
                start_time = max(0, indice)
                end_time = min(tot, indice + 1)
                audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]
                if len(audio_seg) < 1 * nums_per_second:
                    sil = np.zeros(1 * nums_per_second - len(audio_seg), dtype=float)
                    audio_seg = np.concatenate((audio_seg, sil),axis=0)
                audio_seg = torch.from_numpy(audio_seg).unsqueeze(0)
                fbank = preprocess(audio_seg)
                fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
                audio_feature.append(fbank)
            audio_feature = torch.stack(audio_feature,dim=0) # t,L,128
            data.update(
                {
                    'audio':audio_feature,
                    'video':video,
                    'audio_path':audio_path,
                    'video_path':video_path,
                }
            )


        elif task_name == 's4':
            audio_path = sample['audio_path']
            i = sample['idx']
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)
            length = len(audio)
            tot = 5
            nums_per_second = int(length / tot)
            audio_seg = audio[i * nums_per_second : (i+1) * nums_per_second]
            audio_seg = torch.from_numpy(audio_seg).unsqueeze(0)
            fbank = preprocess(audio_seg)
            fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
            data['audio'] = fbank
            data['audio_path'] = audio_path

            ## image
            vpath = sample['image_path']
            image = Image.open(vpath).convert('RGB')
            image = image.resize((224,224))
            image = self.video_processor.preprocess([image],return_tensors='pt')
            image = image['pixel_values']  # t,c,h,w
            data['image'] = image
            data['image_path'] = vpath

            ### mask decoder
            mask_path = sample['mask_path']
            mask = cv2.imread(mask_path)
            gray_mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            gt_mask = gray_mask > 0
            gt_mask = cv2.resize(gt_mask.astype(np.float32),(224,224),interpolation=cv2.INTER_NEAREST)
            gt_mask = torch.from_numpy(gt_mask).unsqueeze(0).to(torch.float32) # (1,1,224,224)
            data['mask'] = gt_mask
            data['mask_path'] = mask_path


        elif task_name == 'ms3':
            audio_path = sample['audio_path']
            i = sample['idx']
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)
            length = len(audio)
            tot = 5
            nums_per_second = int(length / tot)
            audio_seg = audio[i * nums_per_second : (i+1) * nums_per_second]
            audio_seg = torch.from_numpy(audio_seg).unsqueeze(0)
            fbank = preprocess(audio_seg)
            fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
            data['audio'] = fbank
            data['audio_path'] = audio_path

            ## image
            vpath = sample['image_path']
            image = Image.open(vpath).convert('RGB')
            image = image.resize((224,224))
            image = self.video_processor.preprocess([image],return_tensors='pt')
            image = image['pixel_values']  # t,c,h,w
            data['image'] = image
            data['image_path'] = vpath

            
            ### mask decoder
            mask_path = sample['mask_path']
            mask = cv2.imread(mask_path)
            gray_mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            gt_mask = gray_mask > 0
            gt_mask = cv2.resize(gt_mask.astype(np.float32),(224,224),interpolation=cv2.INTER_NEAREST)
            gt_mask = torch.from_numpy(gt_mask).unsqueeze(0).to(torch.float32) # (1,224,224)
            data['mask'] = gt_mask
            data['mask_path'] = mask_path


        elif task_name == 'avss':
            audio_path = sample['audio_path']
            i = sample['idx']
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)
            # if len(audio) < sr: # < 1s
            #     sil = np.zeros(sr-len(audio), dtype=float)
            #     audio = np.concatenate((audio,sil),axis=0)
            tot = 10
            length = len(audio)
            nums_per_second = int(length / tot)
            audio_seg = audio[i * nums_per_second : (i+1) * nums_per_second]
            audio_seg = torch.from_numpy(audio_seg) # L,
            audio_seg = audio_seg.unsqueeze(0)
            fbank = preprocess(audio_seg)
            fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
            data['audio'] = fbank
            data['audio_path'] = audio_path

            vpath = sample['image_path']
            image = Image.open(vpath).convert('RGB')
            image = image.resize((224,224))
            image = self.video_processor.preprocess([image],return_tensors='pt')
            image = image['pixel_values']  # t,c,h,w
            data['image'] = image
            data['image_path'] = vpath
            
            mask_path = sample['mask_path']
            mask = Image.open(mask_path).convert('RGB')
            mask = mask.resize((224,224),Image.Resampling.NEAREST)
            mask = color_mask_to_label(mask,self.v2_pallete) # np.array  (h,w)
            mask = torch.from_numpy(mask).unsqueeze(0).to(torch.long) # (1,224,224)
            data['mask'] = mask
            data['mask_path'] = mask_path


        elif task_name == 'arig':
            audio_path = sample['audio_path']
            i = sample['idx']
            tot = sample['tot']
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)
            length = len(audio)
            tot = 5
            nums_per_second = int(length / tot)
            audio_seg = audio[i * nums_per_second : (i+1) * nums_per_second]
            if len(audio_seg) < 1 * nums_per_second:
                sil = np.zeros(1 * nums_per_second - len(audio_seg), dtype=float)
                audio_seg = np.concatenate((audio_seg, sil),axis=0)
            audio_seg = torch.from_numpy(audio_seg).unsqueeze(0)
            fbank = preprocess(audio_seg)
            fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
            data['audio'] = fbank
            data['audio_path'] = audio_path
            
            vpath = sample['image_path']
            image = Image.open(vpath).convert('RGB')
            image = image.resize((224,224))
            image = self.video_processor.preprocess([image],return_tensors='pt')
            image = image['pixel_values']  # t,c,h,w
            data['image'] = image
            data['image_path'] = vpath

        
        elif task_name == 'ref-avs':
            ## audio
            audio_path = sample['audio_path']
            audio_feature = []
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)
            length = len(audio)
            tot = 10
            nums_per_second = int(length / tot)
            indices = [i for i in range(tot)]
            for indice in indices:
                start_time = max(0, indice)
                end_time = min(tot, indice + 1)
                audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]
                # audio_seg = audio[int(indice*sr):int((indice+1)*sr)]
                # if indice - 0.5 < 0:
                #     sil = np.zeros(2 * nums_per_second - len(audio_seg), dtype=float)
                #     audio_seg = np.concatenate((sil, audio_seg),axis=0)
                # if indice + 1.5 > tot:
                #     sil = np.zeros(2 * nums_per_second - len(audio_seg), dtype=float)
                #     audio_seg = np.concatenate((audio_seg, sil),axis=0)
                if len(audio_seg) < 1 * nums_per_second:
                    sil = np.zeros(1 * nums_per_second - len(audio_seg), dtype=float)
                    audio_seg = np.concatenate((audio_seg, sil),axis=0)
                audio_seg = torch.from_numpy(audio_seg).unsqueeze(0)
                fbank = preprocess(audio_seg)
                fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
                audio_feature.append(fbank)
            audio_feature = torch.stack(audio_feature,dim=0) # t,L,128
            data['audio'] = audio_feature
            data['audio_path'] = audio_path

            ## image
            image_path = sample['image_path']
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224,224))
            image = self.video_processor.preprocess([image],return_tensors='pt')
            image = image['pixel_values']  # t,c,h,w
            data['image'] = image
            data['image_path'] = image_path
            
            ## mask
            mask_path = sample['mask_path']
            mask = cv2.imread(mask_path)
            gray_mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            gt_mask = gray_mask > 0
            gt_mask = cv2.resize(gt_mask.astype(np.float32),(224,224),interpolation=cv2.INTER_NEAREST)
            gt_mask = torch.from_numpy(gt_mask).unsqueeze(0).to(torch.float32) # (1,224,224)
            data['mask'] = gt_mask
            data['mask_path'] = mask_path

        return data


@dataclass
class DataCollatorForUnifiedTestDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        
        tokenizer=self.tokenizer
        batch_input_ids=[]
        batch_label=[]
        batch_X_modals=[]
        batch_metadata=[]
        batch_task_names = []

        for instance in instances:
            instruction = instance['instruction']
            output = instance['output']
            task_name = instance['task_name']
            batch_task_names.append(task_name)

            metadata = {
                'instruction': instruction,
                'output': output,
            }
            
            # if task_name == 'avqa':
            #     question_type = instance.get('question_type',None)
            #     vid = instance.get('vid',None)
            #     qid = instance.get('qid',None)
            #     metadata.update(
            #         {
            #             'question_type':question_type,
            #             'vid':vid,
            #             'qid':qid
            #         }
            #     )
            
            instruction_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(instruction))
            output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))
            
            # if task_name in ['ms3','s4','avss','ref-avs']:
            #     input_ids = instruction_ids + output_ids
            #     label = [-100] * len(instruction_ids) + output_ids
            # else:
            #     input_ids = instruction_ids
            #     label = [-100] * len(instruction_ids)

            input_ids = instruction_ids
            label = [-100] * len(instruction_ids)

            batch_input_ids.append(torch.tensor(input_ids,dtype=torch.long))
            batch_label.append(torch.tensor(label,dtype=torch.long))
            X_modals = {}
            image = instance.get('image',None)
            if image is not None:
                X_modals['<image>'] = image
                metadata['image_path'] = instance.get('image_path','')
                
            video = instance.get('video',None)
            if video is not None:
                X_modals['<video>'] = video
                metadata['video_path'] = instance.get('video_path','')

            audio = instance.get('audio',None)
            if audio is not None:
                X_modals['<audio>'] = audio
                metadata['audio_path'] = instance.get('audio_path','')
            
            mask = instance.get('mask',None)
            if mask is not None:
                X_modals['<mask>'] = mask
                metadata['mask_path'] = instance.get('mask_path','')
            
            batch_X_modals.append(X_modals)
            batch_metadata.append(metadata)

        
        return {
            'batch_input_ids':batch_input_ids,
            'batch_labels':batch_label,
            'batch_X_modals':batch_X_modals,
            'batch_metadata':batch_metadata,
            'batch_task_names':batch_task_names,
        }


def get_dataset_collator(
    data_args,tokenizer: transformers.PreTrainedTokenizer,
    image_processor=None,mode='test',
    image_scale_nums = 2, token_nums_per_scale = 3, test_name = 'test_s', use_process = True,
):
    dataset = UnifiedTestDataset(
        video_processor=image_processor,
        tokenizer=tokenizer,
        avqa_task=data_args.avqa_task,
        ave_task=data_args.ave_task,
        avvp_task = data_args.avvp_task,
        arig_task = data_args.arig_task, 
        avss_task=data_args.avss_task,
        ms3_task=data_args.ms3_task,
        s4_task=data_args.s4_task,
        ref_avs_task=data_args.ref_avs_task,
        test_name=test_name,
        multi_frames=data_args.multi_frames,
        image_scale_nums=image_scale_nums,
        token_nums_per_scale=token_nums_per_scale,
        
    )
    data_collator = DataCollatorForUnifiedTestDataset(tokenizer=tokenizer)

    return dataset,data_collator







