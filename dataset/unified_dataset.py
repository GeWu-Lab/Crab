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


class UnifiedDataset(Dataset):
    def __init__(
        self,
        mode='train', # train,val,test
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
        ref_avs_task = False,
        multi_frames = False,
        image_scale_nums = 2,
        token_nums_per_scale = 3,
        # audio referred image grounding task
        arig_task = False,
        # av caption task
        avcap_task = False,
    ) -> None:
        super().__init__()

        self.mode=mode
        self.video_processor = video_processor
        self.multi_frames = multi_frames
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.video_frame_nums = video_frame_nums

        # if avss_task or ms3_task or s4_task or ref_avs_task:
        token_nums = image_scale_nums * token_nums_per_scale
        mask_token = [f'<mask_{i}>' for i in range(token_nums)]
        self.mask_token = ''.join(mask_token)
        print('mask token: ',self.mask_token)

        self.samples = []
        self.tot = 0

        ### avqa data
        if avqa_task:
            self.add_avqa_task_samples()
        
        ## ave data
        if ave_task:
            self.add_ave_task_samples()
        
        if avvp_task:
            self.add_avvp_task_samples()

        ### ms3 data
        if ms3_task:
            self.add_ms3_task_samples()
        
        ### s4 data
        if s4_task:
            self.add_s4_task_samples()

        if avss_task:
            self.v2_pallete = get_v2_pallete(label_to_idx_path=label_to_idx_path,num_cls=71)
            self.add_avss_task_samples()
        
        if ref_avs_task:
            self.add_ref_avs_samples()

        if arig_task:
            self.add_arig_samples()
        
        if avcap_task:
            self.add_avcap_samples()
        
        print(f'tot training sample nums: {self.tot}')


    def add_avqa_task_samples(self):
        avqa_annotation_path = 'data/music_avqa_data/valid_train_samples.json'
        avqa_data_root = '/group/40061/cserdu/data/music-avqa'
        tot = 0
        with open(avqa_annotation_path,'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_id = sample['video_id']
            question_id = sample['question_id']
            _type = sample['type']
            video_path = sample['video_path']
            audio_path = sample['audio_path']
            question = sample['question']
            answer = sample['answer']
            label_path = join(avqa_data_root,'converted_label',str(question_id)+'.txt')
            output = self.read_label(label_path)
            instruction = f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease answer this question: {question}'
            self.samples.append(
                {
                    'vid':video_id,
                    'qid':question_id,
                    'type':_type,
                    'video_path':video_path,
                    'audio_path':audio_path,
                    # 'question':question,
                    # 'label_path':label_path,
                    'output': output,
                    # 'output':simple_output,
                    'task_name':'avqa',
                    'instruction':instruction,
                }
            )
            tot += 1
        print(f'avqa sample nums: {tot}')
        self.tot += tot


    def add_ave_task_samples(self):
        ave_annotation_path = 'data/ave_data/valid_train_samples.json'
        ave_data_root = '/group/40061/cserdu/data/ave/AVE_Dataset'
        tot = 0
        with open(ave_annotation_path,'r') as f:
            samples = json.load(f)
        for sample in samples:
            event = sample['event']
            vid = sample['vid']
            start_time = sample['start_time']
            end_time = sample['end_time']
            audio_path = join(ave_data_root,'audio_data',vid+'.mp3')
            video_path = join(ave_data_root,'AVE',vid+'.mp4')
            label_path = join(ave_data_root,'converted_label',vid+'.txt')
            output = self.read_label(label_path)
            instruction = f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease describe the events and time range that occurred in the video.'
            # simple_output = f'event:{event} start time:{start_time} end time:{end_time}'
            self.samples.append(
                {
                    'audio_path':audio_path,
                    'video_path':video_path,
                    # 'label_path':label_path,
                    'output': output,
                    # 'output':simple_output,
                    'task_name':'ave',
                    'instruction':instruction,
                }
            )
            tot += 1
        print(f'ave sample nums: {tot}')
        self.tot += tot


    def add_avvp_task_samples(self):
        avvp_annotation_path = 'data/avvp_data/train_samples.json'
        avvp_data_root = '/group/40061/cserdu/data/avvp'
        tot = 0
        with open(avvp_annotation_path,'r') as f:
            samples = json.load(f)
        for sample in samples:
            filename = sample['filename']
            vid = sample['vid']
            event = sample['event']
            label = sample.get('label',None)
            # if label is None or len(label.split(' ')) > 100:
            #     continue
            if label is None:
                continue
            ### use tag token
            label = label.replace('<audio>','<audio_event>')
            label = label.replace('</audio>','</audio_event>')
            label = label.replace('<visual>','<visual_event>')
            label = label.replace('</visual>','</visual_event>')

            audio_path = join(avvp_data_root,'audio_data',vid+'.mp3')
            video_path = join(avvp_data_root,'llp_videos',vid+'.mp4')
            # label_path = join(avvp_data_root,'converted_label',vid+'.txt')
            instruction = f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease determine the events that occur based on the visual and audio information in the video, as well as the start and end times of these events.'
            # simple_output = f'event:{event}'
            self.samples.append(
                {
                    'audio_path':audio_path,
                    'video_path':video_path,
                    # 'label_path':label_path,
                    'output': label,
                    # 'output':simple_output,
                    'task_name':'avvp',
                    'instruction':instruction,
                }
            )
            tot += 1
        print(f'avvp sample nums: {tot}')
        self.tot += tot


    def add_ms3_task_samples(self):
        avs_data_root='/group/40061/cserdu/data/AVS'
        tot = 0
        with open(join(avs_data_root,'ms3_samples.json'),'r') as f:
            samples = json.load(f)
            for sample in samples:
                vid = sample['vid']
                uid = sample['uid']
                s_min = sample['s_min']
                s_sec = sample['s_sec']
                a_obj = sample['a_obj']
                split = sample['split']
                label = sample['label']
                if split != 'train':
                    continue
                audio_path = join(avs_data_root,'v1m',uid,'audio.wav')
                if self.multi_frames:
                    visual_path_list = [join(avs_data_root,'v1m',uid,'frames',f'{i}.jpg') for i in range(5)]
                    mask_path_list = [join(avs_data_root,'v1m',uid,'labels_semantic',f'{i}.png') for i in range(5)]

                    self.samples.append(
                        {
                            'audio':audio_path,
                            'video':visual_path_list,
                            'mask':mask_path_list,
                            'a_obj':a_obj,
                            'instruction':f'This is a video:\n<video_start><video><video_end>\n',
                            'task_name':'ms3'
                        }
                    )
                    tot += 1
                else:
                    image_path_list = [join(avs_data_root,'v1m',uid,'frames',f'{i}.jpg') for i in range(5)]
                    ths = ['first','second','third','fourth','fifth']
                    numbers = list(range(5))
                    for i, th in zip(numbers,ths):
                        visual_path = join(avs_data_root,'v1m',uid,'frames',f'{i}.jpg')
                        mask_path = join(avs_data_root,'v1m',uid,'labels_semantic',f'{i}.png')
                        self.samples.append(
                            {
                                'audio_path':audio_path,
                                'image_path_list':image_path_list,
                                'image_path':visual_path,
                                'mask_path':mask_path,
                                'a_obj':a_obj,
                                'idx':i,
                                'tot':5,
                                'instruction':f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease recognize the category of object making sound in the video, and then segment out the object that makes the sound at the {th} second of the video.',
                                'output':f'The object making the sound in the video is {a_obj}. The mask of the object that makes the sound at the {th} second is <mask_start>{self.mask_token}<mask_end>',
                                'task_name':'ms3',
                            }
                        )
                        tot += 1

        print(f'ms3 sample nums: {tot}')
        self.tot += tot


    def add_s4_task_samples(self):
        avs_data_root='/group/40061/cserdu/data/AVS'
        tot = 0
        with open(join(avs_data_root,'s4_samples.json'),'r') as f:
            samples = json.load(f)
            for sample in samples:
                vid = sample['vid']
                uid = sample['uid']
                s_min = sample['s_min']
                s_sec = sample['s_sec']
                a_obj = sample['a_obj']
                split = sample['split']
                label = sample['label']
                if split != 'train':
                    continue
                image_path_list = [join(avs_data_root,'v1s',uid,'frames',f'{i}.jpg') for i in range(5)]
                audio_path = join(avs_data_root,'v1s',uid,'audio.wav')
                # v1s train only 1 frame
                visual_path = join(avs_data_root,'v1s',uid,'frames','0.jpg')
                mask_path = join(avs_data_root,'v1s',uid,'labels_semantic','0.png')
                self.samples.append(
                    {
                        'audio_path':audio_path,
                        'image_path_list':image_path_list,
                        'image_path':visual_path,
                        'mask_path':mask_path,
                        'a_obj':a_obj,
                        'idx':0,
                        'instruction':f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease recognize the category of object making sound in the video, and then segment out the object that makes the sound at the first second of the video.',
                        'output':f'The object making the sound in the video is {a_obj}. The mask of the object that makes the sound at the first second is <mask_start>{self.mask_token}<mask_end>',
                        'task_name':'s4',
                    }
                )
                tot += 1

        print(f'v1s sample nums: {tot}')
        self.tot += tot


    def add_avss_task_samples(self):
        avs_data_root='/group/40061/cserdu/data/AVS'
        tot = 0
        with open(join(avs_data_root,'avss_samples.json'),'r') as f:
            samples = json.load(f)
            for sample in samples:
                vid = sample['vid']
                uid = sample['uid']
                s_min = sample['s_min']
                s_sec = sample['s_sec']
                a_obj = sample['a_obj']
                split = sample['split']
                label = sample['label']
                if split != 'train':
                    continue
                audio_path = join(avs_data_root,'v2',uid,'audio.wav')
                if self.multi_frames:
                    visual_path_list = [join(avs_data_root,'v1m',uid,'frames',f'{i}.jpg') for i in range(5)]
                    mask_path_list = [join(avs_data_root,'v1m',uid,'labels_semantic',f'{i}.png') for i in range(5)]

                    self.samples.append(
                        {
                            'audio':audio_path,
                            'video':visual_path_list,
                            'mask':mask_path_list,
                            'a_obj':a_obj,
                            'instruction':f'This is a video:\n<video_start><video><video_end>\n',
                            'task_name':'ms3'
                        }
                    )
                    tot += 1
                else:
                    mapping = {
                        1: 'first',
                        2: 'second',
                        3: 'third',
                        4: 'fourth',
                        5: 'fifth',
                        6: 'sixth',
                        7: 'seventh',
                        8: 'eighth',
                        9: 'ninth',
                        10: 'tenth'
                    }
                    # image_path_list = [join(avs_data_root,'v2',uid,'frames',f'{i}.jpg') for i in range(10)]
                    # for i in range(10):
                    #     mask_path = join(avs_data_root,'v2',uid,'labels_rgb',f'{i}.png')
                    #     image_path = join(avs_data_root,'v2',uid,'frames',f'{i}.jpg')
                    #     self.samples.append(
                    #         {
                    #             'audio_path':audio_path,
                    #             'image_path_list':image_path_list,
                    #             'mask_path':mask_path,
                    #             'image_path':image_path,
                    #             'a_obj':a_obj,
                    #             'idx':i,
                    #             'instruction':f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease recognize the category of object making sound in the video, and then segment out the object that makes the sound at the {mapping[i+1]} second of the video.',
                    #             'output':f'The object making the sound in the video is {a_obj}. The mask of the object that makes the sound at the {mapping[i+1]} second is <mask_start>{self.mask_token}<mask_end>',
                    #             'task_name':'avss',  
                    #         }
                    #     )
                    #     tot += 1
                    
                    select_idx = random.randint(0,9)
                    visual_path = join(avs_data_root,'v2',uid,'frames',f'{select_idx}.jpg')
                    image_path_list = [join(avs_data_root,'v2',uid,'frames',f'{i}.jpg') for i in range(10)]
                    mask_path = join(avs_data_root,'v2',uid,'labels_rgb',f'{select_idx}.png')
                    self.samples.append(
                        {
                            'audio_path':audio_path,
                            'image_path':visual_path,
                            'image_path_list':image_path_list,
                            'mask_path':mask_path,
                            'a_obj':a_obj,
                            'idx':select_idx,
                            'instruction':f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease recognize the category of object making sound in the video, and then segment out the object that makes the sound at the {mapping[select_idx+1]} second of the video.',
                            'output':f'The object making the sound in the video is {a_obj}. The mask of the object that makes the sound at the {mapping[select_idx+1]} second is <mask_start>{self.mask_token}<mask_end>',
                            'task_name':'avss',  
                        }
                    )
                    tot += 1

        print(f'avss sample nums: {tot}')
        self.tot += tot


    def get_anchor_point(self,point_list):
        x1, y1 = 1e10, 1e10
        x2, y2 = -1e10, -1e10
        for point in point_list:
            x1 = min(x1,point['x'])
            x2 = max(x2,point['x'])
            y1 = min(y1,point['y'])
            y2 = max(y2,point['y'])
        return x1,y1,x2,y2


    def add_grounded_vqa_samples(self):
        data_root = '/group/40061/cserdu/data/GroundedVQA'
        tot = 0
        with open(join(data_root,'train_grounding.json'),'r') as f:
            data = json.load(f)
        for filename, value in data.items():
            most_common_answer = value['most_common_answer']
            question = value['question']
            width = value['width']
            height = value['height']
            answer_grounding = value['answer_grounding']
            if len(answer_grounding) < 4:
                continue
            x1, y1, x2, y2 = self.get_anchor_point(answer_grounding)
            image_path = join(data_root,'train',filename)
            new_x1 = x1 * 224 / width
            new_y1 = y1 * 224 / height
            new_x2 = x2 * 224 / width
            new_y2 = y2 * 224 / height
            instruction = f'This is an image:\n<image_start><image><image_end>\nQuestion:{question}\nPlease answer the question and output the location information of the answer in the image.'
            output = f'It is {most_common_answer} and position in the image is ({new_x1,new_y1}),({new_x2,new_y2}).'
            self.samples.append(
                {
                    'image_path':image_path,
                    'instruction':instruction,
                    'output':output,
                    'task_name':'grounded_vqa',
                }
            )
            tot += 1
        
        print(f'grounded vqa sample nums: {tot}')


    def add_arig_samples(self,):
        data_root = '/group/40061/cserdu/data/AVS'
        tot = 0
        # vnames = set()
        with open(join(data_root,'v1s_grounding_samples.json'),'r') as f:
            samples = json.load(f)
            for sample in samples:
                split = sample['split']
                if split != 'train':
                    continue
                audio_path = sample['audio_path']
                frame_path = sample['frame_path']
                # vname = frame_path.split('/')[-3]
                # if vname in vnames:
                #     continue
                # vnames.add(vname)
                mask_path = sample['mask_path']
                top_left = sample['top_left']
                a_obj = sample['a_obj']
                x1 = top_left[0]
                y1 = top_left[1]
                bottom_right = sample['bottom_right']
                x2 = bottom_right[0]
                y2 = bottom_right[1]
                if x1 == 1000:  # no sounding object
                    continue
                idx = int(frame_path.split('/')[-1][:-4])
                instruction = f'This is an image:\n<image_start><image><image_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease recognize the category of object that makes the sound and then output its location coordinates.'
                # output = f'The sounding object is {a_obj}. The coordinate of top left corner is <loc_{x1}><loc_{y1}>, the coordinate of bottom right corner is <loc_{x2}><loc_{y2}>'
                # output = f'The sounding object is {a_obj}. Its coordinate is <obj>({x1},{y1}),({x2},{y2})</obj>'
                output = f'The sounding object is {a_obj}. Its coordinate of top left corner is ({x1},{y1}) and coordinate of bottom right corner is ({x2},{y2})'
                self.samples.append(
                    {
                        'audio_path':audio_path,
                        'image_path':frame_path,
                        'mask_path':mask_path,
                        'idx':idx,
                        'tot':5,
                        # 'top_left':top_left,
                        # 'bottom_right':bottom_right,
                        'task_name':'arig',
                        'instruction':instruction,
                        'output':output,
                    }
                )
                tot += 1
        
        # vnames = set()
        # with open(join(data_root,'v2_grounding_samples.json'),'r') as f:
        #     samples = json.load(f)
        #     for sample in samples:
        #         split = sample['split']
        #         if split != 'train':
        #             continue
        #         audio_path = sample['audio_path']
        #         frame_path = sample['frame_path']
        #         vname = frame_path.split('/')[-3]
        #         if vname in vnames:
        #             continue
        #         vnames.add(vname)
        #         mask_path = sample['mask_path']
        #         value2point = sample['value2point']
        #         coordinate = ''
        #         for value, point_list in value2point.items():
        #             x1 = point_list[0][0]
        #             y1 = point_list[0][1]
        #             x2 = point_list[1][0]
        #             y2 = point_list[1][1]
        #             coordinate += f'<obj>({x1},{y1})({x2},{y2})</obj>'
        #         idx = int(frame_path.split('/')[-1][:-4])
        #         instruction = f'This is an image:\n<image_start><image><image_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nWhat are the coordinates of the object making the sound in the image?'
        #         output = f'The coordinates are {coordinate}.'
        #         self.samples.append(
        #             {
        #                 'audio_path':audio_path,
        #                 'image_path':frame_path,
        #                 'mask_path':mask_path,
        #                 'idx':idx,
        #                 'tot':10,
        #                 'task_name':'arig',
        #                 'instruction':instruction,
        #                 'output':output,
        #             }
        #         )
        #         tot += 1
        
        print(f'audio referred image grounding sample nums: {tot}')
        self.tot += tot


    def add_avcap_samples(self):
        data_root = '/group/40061/cserdu/data/valor'
        tot = 0
        with open(join(data_root,'train_samples.json'),'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_id = sample['video_id']
            desc = sample['desc']
            video_path = join(data_root,'video_data',video_id+'.mp4')
            audio_path = join(data_root,'audio_data',video_id+'.mp3')
            instruction = f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease describe this video and audio.'
            self.samples.append(
                {
                    'audio_path':audio_path,
                    'video_path':video_path,
                    'instruction':instruction,
                    'output':desc,
                    'task_name':'avcap',
                }
            )
            tot += 1
        self.tot += tot
        print(f'avcap sample nums: {tot}')


    def add_ref_avs_samples(self):
        data_root = '/group/40061/cserdu/data/ref-avs/REFAVS'
        tot = 0
        with open(join(data_root,'metadata.csv'),'r') as f:
            rows = csv.reader(f)
            for row in rows:
                vid, uid, split, fid, exp = row
                if split != 'train':
                    continue
                vid = uid.rsplit('_', 2)[0]  # TODO: use encoded id.
                obj = uid.rsplit('_',2)[1]
                audio_path = join(data_root,'media',vid,'audio.wav')
                image_path_list = [join(data_root,'media',vid,'frames',str(i)+'.jpg') for i in range(10)]
                mapping = {
                    1: 'first',
                    2: 'second',
                    3: 'third',
                    4: 'fourth',
                    5: 'fifth',
                    6: 'sixth',
                    7: 'seventh',
                    8: 'eighth',
                    9: 'ninth',
                    10: 'tenth'
                }
                ### for avs finetune
                # for i in range(10):
                #     image_path = join(data_root,'media',vid,'frames',str(i)+'.jpg')
                #     mask_path = join(data_root,'gt_mask',vid,'fid_'+str(fid),'0000'+str(i)+'.png')
                #     instruction = f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease segment out {exp.lower()} at the {mapping[i+1]} second in the video.'
                #     output = f'At the {mapping[i+1]} second in the video, {exp} is {obj}. Its mask is <mask_start>{self.mask_token}<mask_end>'
                #     self.samples.append(
                #         {
                #             'instruction': instruction,
                #             'output': output,
                #             'image_path':image_path,
                #             'mask_path':mask_path,
                #             'audio_path':audio_path,
                #             'image_path_list':image_path_list,
                #             'vid':vid,
                #             'uid':uid,
                #             'fid':fid,
                #             'task_name':'ref-avs',
                #         }
                #     )
                #     tot += 1
                ### for unified finetune
                select_idx = random.randint(0,9)
                image_path = join(data_root,'media',vid,'frames',str(select_idx)+'.jpg')
                mask_path = join(data_root,'gt_mask',vid,'fid_'+str(fid),'0000'+str(select_idx)+'.png')
                instruction = f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease segment out {exp.lower()} at the {mapping[select_idx+1]} second in the video.'
                output = f'At the {mapping[select_idx+1]} second in the video, {exp} is {obj}. Its mask is <mask_start>{self.mask_token}<mask_end>'
                self.samples.append(
                    {
                        'instruction': instruction,
                        'output': output,
                        'image_path':image_path,
                        'mask_path':mask_path,
                        'audio_path':audio_path,
                        'image_path_list':image_path_list,
                        'vid':vid,
                        'uid':uid,
                        'fid':fid,
                        'task_name':'ref-avs',
                    }
                )
                tot += 1
        
        self.tot += tot
        print(f'ref-avs sample nums: {tot}')


    def read_label(self,label_path):
        with open(label_path,'r') as f:
            label = f.read()
        return label


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
            'instruction':instruction,
            'output':output,
            'task_name':task_name,
        }
        
        if task_name == 'avqa':
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
            
            ### process audio
            audio_feature = []
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)
            # length = len(audio)
            # tot = 10
            # nums_per_second = length / tot
            # audio = audio[ : int(tot * nums_per_second)]
            # audio = torch.from_numpy(audio).unsqueeze(0)
            # fbank = preprocess(audio)
            # fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
            # audio_feature = fbank.reshape(tot,-1,fbank.shape[-1])

            length = len(audio)
            tot = 10
            indices = [i for i in range(tot)]
            nums_per_second = int(length / tot)
            # audio = audio[: tot * nums_per_second]
            # if len(audio) < tot * nums_per_second:
            #     sil = np.zeros(tot * nums_per_second - len(audio), dtype=float)
            #     audio = np.concatenate((audio, sil),axis=0)
            # audio = torch.from_numpy(audio) # L,
            # audio = audio.unsqueeze(0)
            # audio = preprocess(audio)
            # audio = audio.to(torch.float32) # 1,L,128   T=1
            # data['audio'] = audio
            
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
            data['video'] = video
            
            ### process audio
            audio_feature = []
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)
            length = len(audio)
            tot = 10
            nums_per_second = int(length / tot)
            # audio = audio[ : int(tot * nums_per_second)]
            # audio = torch.from_numpy(audio).unsqueeze(0)
            # fbank = preprocess(audio)
            # fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
            # audio_feature = fbank.reshape(tot,-1,fbank.shape[-1])

            # length = len(audio)
            # tot = 10
            # nums_per_second = int(length / tot)
            # audio = audio[: tot * nums_per_second]
            # if len(audio) < tot * nums_per_second:
            #     sil = np.zeros(tot * nums_per_second - len(audio), dtype=float)
            #     audio = np.concatenate((audio, sil),axis=0)
            # audio = torch.from_numpy(audio) # L,
            # audio = audio.unsqueeze(0)
            # audio = preprocess(audio)
            # audio = audio.to(torch.float32) # 1,L,128   T=1
            # data['audio'] = audio

            indices = [i for i in range(tot)]
            for indice in indices:
                start_time = max(0, indice)
                end_time = min(tot, indice + 1)
                audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]
                # audio_seg = audio[int(indice*sr):int((indice+1)*sr)]
                # if indice < 0:
                #     sil = np.zeros(1 * nums_per_second - len(audio_seg), dtype=float)
                #     audio_seg = np.concatenate((sil, audio_seg),axis=0)
                # if indice + 1 > tot:
                #     sil = np.zeros(1 * nums_per_second - len(audio_seg), dtype=float)
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
        
        elif task_name == 'ms3' or task_name == 's4':
            # a_obj = sample['a_obj']

            if self.multi_frames:
                audio_path = sample['audio']
                audio, sr = librosa.load(audio,sr=16000,mono=True)
                audio = torch.from_numpy(audio).to(torch.float32) # L,

                video = []
                visual_path_list = sample['video']
                for vpath in visual_path_list:
                    image = Image.open(vpath).convert('RGB')
                    image = image.resize((224,224))
                    image = self.video_processor.preprocess([image],return_tensors='pt')
                    image = image['pixel_values']  # t,c,h,w
                    video.append(image)
                video = torch.cat(video,dim=0) # t,c,h,w

                masks = []
                mask_path_list = sample['mask']
                for mask_path in mask_path_list:
                    mask = cv2.imread(mask_path)
                    gray_mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
                    gt_mask = gray_mask > 0
                    gt_mask = cv2.resize(gt_mask.astype(np.float32),(224,224),interpolation=cv2.INTER_NEAREST)
                    gt_mask = torch.from_numpy(gt_mask).unsqueeze(0).to(torch.float32) # (1,224,224)
                    masks.append(gt_mask)
                masks = torch.stack(masks,dim=0) # t,1,h,w
            else:
                ## audio
                audio_path = sample['audio_path']
                i = sample['idx']
                audio, sr = librosa.load(audio_path,sr=16000,mono=True)
                length = len(audio)
                tot = 5
                nums_per_second = int(length / tot)
                indices = [i for i in range(tot)]
                audio_feature = []
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

                # if len(audio) < sr: # < 1s
                #     sil = np.zeros(sr-len(audio), dtype=float)
                #     audio = np.concatenate((audio,sil),axis=0)
                # start_time = max(0, i)
                # end_time = min(tot, i + 1)
                # audio = audio[int(start_time * nums_per_second) : int(end_time * nums_per_second)]
                # audio = audio[: tot * nums_per_second]
                # if len(audio) < tot * nums_per_second:
                #     sil = np.zeros(tot * nums_per_second - len(audio), dtype=float)
                #     audio = np.concatenate((audio, sil),axis=0)
                # audio = torch.from_numpy(audio) # L,
                # audio = audio.unsqueeze(0)
                # audio = preprocess(audio)
                # audio = audio.to(torch.float32) # 1,L,128   T=1
                # data['audio'] = audio

                ## image
                # vpath = sample['image_path']
                # image = Image.open(vpath).convert('RGB')
                # image = image.resize((224,224))
                # image = self.video_processor.preprocess([image],return_tensors='pt')
                # image = image['pixel_values']  # t,c,h,w
                # data['image'] = image

                ## video
                image_path_list = sample['image_path_list']
                video = []
                for vpath in image_path_list:
                    image = Image.open(vpath).convert('RGB')
                    image = image.resize((224,224))
                    image = self.video_processor.preprocess([image],return_tensors='pt')
                    image = image['pixel_values']  # t,c,h,w
                    video.append(image)
                video = torch.cat(video,dim=0) # t,c,h,w
                data['video'] = video
                
                ### mask decoder
                # mask_path = sample['mask_path']
                # mask = cv2.imread(mask_path)
                # gray_mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
                # gt_mask = gray_mask > 0
                # gt_mask = cv2.resize(gt_mask.astype(np.float32),(224,224),interpolation=cv2.INTER_NEAREST)
                # gt_mask = torch.from_numpy(gt_mask).unsqueeze(0).to(torch.float32) # (1,1,224,224)
                # data['mask'] = gt_mask

                ### vqgan
                # mask = cv2.imread(mask_path)
                # gray_mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
                # gt_mask = gray_mask > 0
                # gt_mask = cv2.resize(gt_mask.astype(np.float32),(224,224),interpolation=cv2.INTER_NEAREST)
                # gt_mask = torch.from_numpy(gt_mask).unsqueeze(0).to(torch.float32) # (1,224,224)
                # gt_mask = gt_mask * 255.
                # gt_mask = gt_mask.repeat(3,1,1) # 3,h,w
                # gt_mask = gt_mask / 127.5 -1

                # data.update(
                #     {
                #         'audio':audio,
                #         'image':image,
                #         'mask':gt_mask,
                #     }
                # )

        elif task_name == 'avss':
            if self.multi_frames:
                pass
            else:
                ## audio
                audio_path = sample['audio_path']
                i = sample['idx']
                audio, sr = librosa.load(audio_path,sr=16000,mono=True)
                # if len(audio) < sr: # < 1s
                #     sil = np.zeros(sr-len(audio), dtype=float)
                #     audio = np.concatenate((audio,sil),axis=0)
                tot = 10
                length = len(audio)
                nums_per_second = int(length / tot)
                indices = [i for i in range(tot)]
                audio_feature = []
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

                # start_time = max(0, i)
                # end_time = min(tot, i + 1)
                # audio = audio[int(start_time * nums_per_second) : int(end_time * nums_per_second)]
                # # audio = audio[: tot * nums_per_second]
                # # if len(audio) < tot * nums_per_second:
                # #     sil = np.zeros(tot * nums_per_second - len(audio), dtype=float)
                # #     audio = np.concatenate((audio, sil),axis=0)
                # audio = torch.from_numpy(audio) # L,
                # audio = audio.unsqueeze(0)
                # audio = preprocess(audio)
                # audio = audio.to(torch.float32) # 1,L,128   T=1
                # data['audio'] = audio

                ## video
                video = []
                image_path_list = sample['image_path_list']
                for vpath in image_path_list:
                    image = Image.open(vpath).convert('RGB')
                    image = image.resize((224,224))
                    image = self.video_processor.preprocess([image],return_tensors='pt')
                    image = image['pixel_values']  # t,c,h,w
                    video.append(image)
                video = torch.cat(video,dim=0)
                data['video'] = video

                ## image
                # vpath = sample['image_path']
                # image = Image.open(vpath).convert('RGB')
                # image = image.resize((224,224))
                # image = self.video_processor.preprocess([image],return_tensors='pt')
                # image = image['pixel_values']  # t,c,h,w
                # data['image'] = image
                
                ## mask
                # mask_path = sample['mask_path']
                # mask = Image.open(mask_path).convert('RGB')
                # mask = mask.resize((224,224),Image.Resampling.NEAREST)
                # mask = color_mask_to_label(mask,self.v2_pallete) # np.array  (h,w)
                # mask = torch.from_numpy(mask).unsqueeze(0).to(torch.long) # (1,224,224)
                # data['mask'] = mask

                # data.update(
                #     {
                #         'audio':audio,
                #         'image':image,
                #         'mask':mask,
                #     }
                # )

        elif task_name == 'grounded_vqa':
            image_path = sample['image_path']
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224,224))
            image = self.video_processor.preprocess([image],return_tensors='pt')
            image = image['pixel_values']  # t,c,h,w
            data['image'] = image
        
        elif task_name == 'arig':
            audio_path = sample['audio_path']
            i = sample['idx']
            tot = sample['tot']
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)
            # if len(audio) < sr: # < 1s
            #     sil = np.zeros(sr-len(audio), dtype=float)
            #     audio = np.concatenate((audio,sil),axis=0)
            # length = len(audio)
            # nums_per_second = length / tot
            # start_time = max(0, i)
            # end_time = min(tot, i + 1)
            # audio = audio[int(nums_per_second * start_time) : int(nums_per_second * end_time)]
            length = len(audio)
            tot = 5
            nums_per_second = int(length / tot)
            indices = [i for i in range(tot)]
            audio_feature = []
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
            
            # audio = audio[: tot * nums_per_second]
            # if len(audio) < tot * nums_per_second:
            #     sil = np.zeros(tot * nums_per_second - len(audio), dtype=float)
            #     audio = np.concatenate((audio, sil),axis=0)
            # audio = torch.from_numpy(audio) # L,
            # audio = audio.unsqueeze(0)
            # audio = preprocess(audio)
            # audio = audio.to(torch.float32) # 1,L,128   T=1
            # data['audio'] = audio

            vpath = sample['image_path']
            image = Image.open(vpath).convert('RGB')
            image = image.resize((224,224))
            image = self.video_processor.preprocess([image],return_tensors='pt')
            image = image['pixel_values']  # t,c,h,w
            data['image'] = image

            # data.update(
            #     {
            #         'audio':audio,
            #         'image':image,
            #     }
            # )

        elif task_name == 'avcap':
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
            
            ### process audio
            audio_feature = []
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)
            length = len(audio)
            # indices = [i for i in range(10)]
            tot = 10
            nums_per_second = int(length / tot)
            # audio = audio[: tot * nums_per_second]
            # if len(audio) < tot * nums_per_second:
            #     sil = np.zeros(tot * nums_per_second - len(audio), dtype=float)
            #     audio = np.concatenate((audio, sil),axis=0)
            # audio = torch.from_numpy(audio) # L,
            # audio = audio.unsqueeze(0)
            # audio = preprocess(audio)
            # audio = audio.to(torch.float32) # 1,L,128   T=1
            # data['audio'] = audio

            indices = [i for i in range(tot)]
            audio_feature = []
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

        elif task_name == 'ref-avs':
            ## video
            image_path_list = sample['image_path_list']
            video = []
            for path in image_path_list:
                image = Image.open(path).convert('RGB')
                image = image.resize((224,224))
                image = self.video_processor.preprocess([image],return_tensors='pt')
                image = image['pixel_values']  # t,c,h,w
                video.append(image)
            video = torch.cat(video,dim=0)
            data['video'] = video

            ## audio
            audio_path = sample['audio_path']
            audio_feature = []
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)
            length = len(audio)
            tot = 10
            nums_per_second = int(length / tot)
            # audio = audio[: tot * nums_per_second]
            # if len(audio) < tot * nums_per_second:
            #     sil = np.zeros(tot * nums_per_second - len(audio), dtype=float)
            #     audio = np.concatenate((audio, sil),axis=0)
            # audio = torch.from_numpy(audio) # L,
            # audio = audio.unsqueeze(0)
            # audio = preprocess(audio)
            # audio = audio.to(torch.float32) # 1,L,128   T=1
            # data['audio'] = audio

            indices = [i for i in range(tot)]
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

            ## image
            # image_path = sample['image_path']
            # image = Image.open(image_path).convert('RGB')
            # image = image.resize((224,224))
            # image = self.video_processor.preprocess([image],return_tensors='pt')
            # image = image['pixel_values']  # t,c,h,w
            # data['image'] = image
            
            ## mask
            # mask_path = sample['mask_path']
            # mask = cv2.imread(mask_path)
            # gray_mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            # gt_mask = gray_mask > 0
            # gt_mask = cv2.resize(gt_mask.astype(np.float32),(224,224),interpolation=cv2.INTER_NEAREST)
            # gt_mask = torch.from_numpy(gt_mask).unsqueeze(0).to(torch.float32) # (1,224,224)
            # data['mask'] = gt_mask

        return data


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
        # avcap task
        avcap_task = False,
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

        ### avqa data
        if avqa_task:
            self.add_avqa_task_samples()

        ### ave data
        if ave_task:
            self.add_ave_task_samples()

        if avvp_task:
            self.add_avvp_task_samples()

        ### ms3 data
        if ms3_task:
            self.add_ms3_task_samples()
        
        ### s4 data
        if s4_task:
            self.add_s4_task_samples()

        if avss_task:
            self.v2_pallete = get_v2_pallete(label_to_idx_path=label_to_idx_path,num_cls=71)
            self.add_avss_task_samples()
        
        if ref_avs_task:
            self.add_ref_avs_samples()
        
        if arig_task:
            self.add_arig_samples()
        
        if avcap_task:
            self.add_avcap_samples()
        
        print(f'tot test sample nums: {self.tot}')


    def add_avqa_task_samples(self):
        avqa_annotation_path = 'data/music_avqa_data/test_samples.json'
        avqa_data_root = '/group/40061/cserdu/data/music-avqa'
        tot = 0
        with open(avqa_annotation_path,'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_id = sample['video_id']
            question_id = sample['question_id']
            questio_type = sample['type']
            video_path = sample['video_path']
            audio_path = sample['audio_path']
            question = sample['question']
            answer = sample['answer']
            instruction = f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease answer this question: {question}'
            self.samples.append(
                {
                    'vid':video_id,
                    'qid':question_id,
                    'question_type':questio_type,
                    'video_path':video_path,
                    'audio_path':audio_path,
                    'question':question,
                    'task_name':'avqa',
                    'instruction':instruction,
                    'output': answer,
                }
            )
            tot += 1
        print(f'avqa sample nums: {tot}')


    def add_ave_task_samples(self):
        ave_annotation_path = 'data/ave_data/test_samples.json'
        ave_data_root = '/group/40061/cserdu/data/ave/AVE_Dataset'
        tot = 0
        with open(ave_annotation_path,'r') as f:
            samples = json.load(f)
        for sample in samples:
            event = sample['event']
            vid = sample['vid']
            start_time = sample['start_time']
            end_time = sample['end_time']
            audio_path = join(ave_data_root,'audio_data',vid+'.mp3')
            video_path = join(ave_data_root,'AVE',vid+'.mp4')
            instruction = f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease describe the events and time range that occurred in the video.'
            self.samples.append(
                {
                    'audio_path':audio_path,
                    'video_path':video_path,
                    'task_name':'ave',
                    'instruction':instruction,
                    'output': f'event:{event} start_time:{start_time} end_time:{end_time}'
                }
            )
            tot += 1
        print(f'ave sample nums: {tot}')


    def add_avvp_task_samples(self):
        avvp_annotation_path = 'data/avvp_data/test_samples.json'
        avvp_data_root = '/group/40061/cserdu/data/avvp'
        tot = 0
        with open(avvp_annotation_path,'r') as f:
            samples = json.load(f)
        for sample in samples:
            filename = sample['filename']
            vid = sample['vid']
            event = sample['event']
            audio_path = join(avvp_data_root,'audio_data',vid+'.mp3')
            video_path = join(avvp_data_root,'llp_videos',vid+'.mp4')
            instruction = f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease determine the events that occur based on the visual and audio information in the video, as well as the start and end times of these events.'
            self.samples.append(
                {
                    'audio_path':audio_path,
                    'video_path':video_path,
                    'output': event,
                    'task_name':'avvp',
                    'instruction':instruction,
                }
            )
            tot += 1
        print(f'avvp sample nums: {tot}')
        self.tot += tot


    def add_ms3_task_samples(self):
        avs_data_root='/group/40061/cserdu/data/AVS'
        tot = 0
        with open(join(avs_data_root,'ms3_samples.json'),'r') as f:
            samples = json.load(f)
            for sample in samples:
                vid = sample['vid']
                uid = sample['uid']
                s_min = sample['s_min']
                s_sec = sample['s_sec']
                a_obj = sample['a_obj']
                split = sample['split']
                label = sample['label']
                if split != 'test':
                    continue
                audio_path = join(avs_data_root,'v1m',uid,'audio.wav')
                if self.multi_frames:
                    visual_path_list = [join(avs_data_root,'v1m',uid,'frames',f'{i}.jpg') for i in range(5)]
                    mask_path_list = [join(avs_data_root,'v1m',uid,'labels_semantic',f'{i}.png') for i in range(5)]

                    self.samples.append(
                        {
                            'audio':audio_path,
                            'video':visual_path_list,
                            'mask':mask_path_list,
                            'a_obj':a_obj,
                            'instruction':f'This is a video:\n<video_start><video><video_end>\n',
                            'task_name':'ms3'
                        }
                    )
                    tot += 1
                else:
                    # for i in range(5):
                    #     visual_path = join(avs_data_root,'v1m',uid,'frames',f'{i}.jpg')
                    #     mask_path = join(avs_data_root,'v1m',uid,'labels_semantic',f'{i}.png')
                    #     self.samples.append(
                    #         {
                    #             'audio_path':audio_path,
                    #             'image_path':visual_path,
                    #             'mask_path':mask_path,
                    #             'a_obj':a_obj,
                    #             'idx':i,
                    #             'tot':5,
                    #             'instruction':f'This is an image:\n<image_start><image><image_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease segment out the object that makes the sound in the image.',
                    #             'output':f'The object that makes the sound is <mask_start>{self.mask_token}<mask_end>',
                    #             'task_name':'ms3',
                    #         }
                    #     )
                    #     tot += 1
                    image_path_list = [join(avs_data_root,'v1m',uid,'frames',f'{i}.jpg') for i in range(5)]
                    for i in range(5):
                        visual_path = join(avs_data_root,'v1m',uid,'frames',f'{i}.jpg')
                        mask_path = join(avs_data_root,'v1m',uid,'labels_semantic',f'{i}.png')
                        self.samples.append(
                            {
                                'audio_path':audio_path,
                                'image_path_list':image_path_list,
                                'image_path':visual_path,
                                'mask_path':mask_path,
                                'a_obj':a_obj,
                                'idx':i,
                                'tot':5,
                                'instruction':f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease recognize the category of object making sound in the video, and then segment out the object that makes the sound at the third second of the video.',
                                'output':f'The object making the sound in the video is {a_obj}. The mask of the object that makes the sound in the third second is <mask_start>{self.mask_token}<mask_end>',
                                'task_name':'ms3',
                            }
                        )
                        tot += 1
        print(f'ms3 sample nums: {tot}')


    def add_s4_task_samples(self):
        avs_data_root='/group/40061/cserdu/data/AVS'
        tot = 0
        with open(join(avs_data_root,'s4_samples.json'),'r') as f:
            samples = json.load(f)
            for sample in samples:
                vid = sample['vid']
                uid = sample['uid']
                s_min = sample['s_min']
                s_sec = sample['s_sec']
                a_obj = sample['a_obj']
                split = sample['split']
                label = sample['label']
                if split != 'test':
                    continue
                audio_path = join(avs_data_root,'v1s',uid,'audio.wav')
                # v1s test for 5 frames
                image_path_list = [join(avs_data_root,'v1s',uid,'frames',f'{i}.jpg') for i in range(5)]
                ths = ['first','second','third','fourth','fifth']
                numbers = list(range(5))
                for i, th in zip(numbers,ths):
                    visual_path = join(avs_data_root,'v1s',uid,'frames',f'{i}.jpg')
                    mask_path = join(avs_data_root,'v1s',uid,'labels_semantic',f'{i}.png')
                    self.samples.append(
                        {
                            'audio_path':audio_path,
                            'image_path_list':image_path_list,
                            'image_path':visual_path,
                            'mask_path':mask_path,
                            'a_obj':a_obj,
                            'idx':i,
                            'tot':5,
                            'instruction':f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease recognize the category of object making sound in the video, and then segment out the object that makes the sound at the {th} second of the video.',
                            'output':f'The object making the sound in the video is {a_obj}. The mask of the object that makes the sound at the {th} second is <mask_start>{self.mask_token}<mask_end>',
                            'task_name':'s4',
                        }
                    )
                    tot += 1
        print(f'v1s sample nums: {tot}')


    def add_avss_task_samples(self):
        avs_data_root='/group/40061/cserdu/data/AVS'
        tot = 0
        with open(join(avs_data_root,'avss_samples.json'),'r') as f:
            samples = json.load(f)
            for sample in samples:
                vid = sample['vid']
                uid = sample['uid']
                s_min = sample['s_min']
                s_sec = sample['s_sec']
                a_obj = sample['a_obj']
                split = sample['split']
                label = sample['label']
                if split != 'test':
                    continue
                audio_path = join(avs_data_root,'v2',uid,'audio.wav')
                if self.multi_frames:
                    visual_path_list = [join(avs_data_root,'v1m',uid,'frames',f'{i}.jpg') for i in range(5)]
                    mask_path_list = [join(avs_data_root,'v1m',uid,'labels_semantic',f'{i}.png') for i in range(5)]

                    self.samples.append(
                        {
                            'audio':audio_path,
                            'video':visual_path_list,
                            'mask':mask_path_list,
                            'a_obj':a_obj,
                            'instruction':f'This is a video:\n<video_start><video><video_end>\n',
                            'task_name':'ms3'
                        }
                    )
                    tot += 1
                else:
                    for i in range(10):
                        visual_path = join(avs_data_root,'v2',uid,'frames',f'{i}.jpg')
                        mask_path = join(avs_data_root,'v2',uid,'labels_rgb',f'{i}.png')
                        self.samples.append(
                            {
                                'audio_path':audio_path,
                                'image_path':visual_path,
                                'mask_path':mask_path,
                                'a_obj':a_obj,
                                'idx':i,
                                # 'instruction':f'This is an image:\n<image_start><image><image_end>\nWhere is <audio_start><audio><audio_end> in the image?',
                                'instruction':f'This is an image:\n<image_start><image><image_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease recognize the category of the object making the sound in the image and segment it out.',
                                'output':f'The category of the sound-making object and its mask image are <mask_start>{self.mask_token}<mask_end>',
                                'task_name':'avss',
                                # 'visual_question':'Segmentation task. Please describe this image.',
                                # 'audio_question':'Segmentation task. Please describe this audio.',
                            }
                        )
                        tot += 1
        print(f'avss sample nums: {tot}')


    def add_arig_samples(self,):
        data_root = '/group/40061/cserdu/data/AVS'
        tot = 0
        with open(join(data_root,'v1s_grounding_samples.json'),'r') as f:
            samples = json.load(f)
            for sample in samples:
                split = sample['split']
                if split != 'test':
                    continue
                audio_path = sample['audio_path']
                frame_path = sample['frame_path']
                mask_path = sample['mask_path']
                top_left = sample['top_left']
                bottom_right = sample['bottom_right']
                a_obj = sample['a_obj']
                x1, y1 = top_left
                x2, y2 = bottom_right
                if x1 == 1000:
                    continue # no sounding object
                idx = int(frame_path.split('/')[-1][:-4])
                instruction = f'This is an image:\n<image_start><image><image_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease recognize the category of object that makes the sound and then output the location coordinates.'
                output = f'The sounding object is {a_obj}. Its coordinates are <obj>({x1},{y1})({x2},{y2})</obj>.'
                self.samples.append(
                    {
                        'audio_path':audio_path,
                        'image_path':frame_path,
                        'mask_path':mask_path,
                        'idx':idx,
                        'tot':5,
                        'task_name':'arig',
                        'instruction':instruction,
                        'output':output,
                    }
                )
                tot += 1
        
        # with open(join(data_root,'v2_grounding_samples.json'),'r') as f:
        #     samples = json.load(f)
        #     for sample in samples:
        #         split = sample['split']
        #         if split != 'test':
        #             continue
        #         audio_path = sample['audio_path']
        #         frame_path = sample['frame_path']
        #         mask_path = sample['mask_path']
        #         value2point = sample['value2point']
        #         coordinate = ''
        #         for value, point_list in value2point.items():
        #             coordinate += f'<obj>({point_list[0][0]},{point_list[0][1]})({point_list[1][0]},{point_list[1][1]})</obj>'
        #         idx = int(frame_path.split('/')[-1][:-4])
        #         instruction = f'This is an image:\n<image_start><image><image_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nWhat are the coordinates of the object making the sound in the image?'
        #         output = f'The coordinates are {coordinate}.'
        #         self.samples.append(
        #             {
        #                 'audio_path':audio_path,
        #                 'image_path':frame_path,
        #                 'mask_path':mask_path,
        #                 'idx':idx,
        #                 'tot':10,
        #                 'task_name':'arig',
        #                 'instruction':instruction,
        #                 'output':output,
        #             }
        #         )
        #         tot += 1
    
        print(f'audio referred image grounding sample nums: {tot}')
        self.tot += tot


    def add_avs_bench_samples(self):
        data_root = '/group/40061/cserdu/data/AVS'
        with open('/group/40061/cserdu/Unified/SSLalignment/metadata/s4_box.json','r') as f:
            samples = json.load(f)
            for sample in samples:
                image = sample['image']
                audio = sample['audio']
                

    def add_avcap_samples(self):
        data_root = '/group/40061/cserdu/data/valor'
        tot = 0
        with open(join(data_root,'val_samples.json'),'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_id = sample['video_id']
            desc = sample['desc']
            video_path = join(data_root,'video_data',video_id+'.mp4')
            audio_path = join(data_root,'audio_data',video_id+'.mp3')
            instruction = f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease describe this video.'
            self.samples.append(
                {
                    'audio_path':audio_path,
                    'video_path':video_path,
                    'instruction':instruction,
                    'output':desc,
                    'task_name':'avcap',
                }
            )
            tot += 1
        self.tot += tot
        print(f'avcap sample nums: {tot}')


    def add_ref_avs_samples(self):
        data_root = '/group/40061/cserdu/data/ref-avs/REFAVS'
        tot = 0
        with open(join(data_root,'metadata.csv'),'r') as f:
            rows = csv.reader(f)
            for row in rows:
                vid, uid, split, fid, exp = row
                if split != self.test_name:  # test_s,test_u,test_n
                    continue
                vid = uid.rsplit('_', 2)[0]  # TODO: use encoded id.
                audio_path = join(data_root,'media',vid,'audio.wav')
                image_path_list = [join(data_root,'media',vid,'frames',str(i)+'.jpg') for i in range(10)]
                for i in range(10):
                    image_path = join(data_root,'media',vid,'frames',str(i)+'.jpg')
                    mask_path = join(data_root,'gt_mask',vid,'fid_'+str(fid),'0000'+str(i)+'.png')
                    instruction = f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nThis is an image:\n<image_start><image><image_end>\nPlease segment out the corresponding object in the image based on the referential expression: {exp}'
                    # output = f'The object that makes the sound is <mask_start>{self.mask_token}<mask_end>'
                    output = f'{exp} is <mask_start>{self.mask_token}<mask_end>'
                    self.samples.append(
                        {
                            'instruction': instruction,
                            'output': output,
                            'image_path':image_path,
                            'mask_path':mask_path,
                            'audio_path':audio_path,
                            'image_path_list':image_path_list,
                            'vid':vid,
                            'uid':uid,
                            'fid':fid,
                            'task_name':'ref-avs',
                        }
                    )
                    tot += 1
        
        self.tot += tot
        print(f'ref-avs {self.test_name} sample nums: {tot}')



    def __len__(self):
        return len(self.samples)


    def read_label(self,label_path):
        if not os.path.exists(label_path):
            return 'no label.'
        with open(label_path,'r') as f:
            label = f.read()
        return label


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

            question_type = sample['question_type']
            vid = sample['vid']
            qid = sample['qid']
            data['question_type'] = question_type
            data['vid'] = vid
            data['qid'] = qid

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
            # length = len(audio)
            # tot = 10
            # nums_per_second = length / tot
            # audio = audio[ : int(tot * nums_per_second)]
            # audio = torch.from_numpy(audio).unsqueeze(0)
            # fbank = preprocess(audio)
            # fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
            # audio_feature = fbank.reshape(tot,-1,fbank.shape[-1])

            length = len(audio)
            tot = 10
            nums_per_second = int(length / tot)
            indices = [i for i in range(10)]
            for indice in indices:
                start_time = max(0, indice)
                end_time = min(tot, indice + 1)
                audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]
                # if indice < 0:
                #     sil = np.zeros(2 * nums_per_second - len(audio_seg), dtype=float)
                #     audio_seg = np.concatenate((sil, audio_seg),axis=0)
                if indice + 1 > tot:
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

        elif task_name == 'ms3' or task_name == 's4':
            
            if self.multi_frames:
                audio_path = sample['audio']
                audio, sr = librosa.load(audio,sr=16000,mono=True)
                audio = torch.from_numpy(audio).to(torch.float32) # L,

                video = []
                visual_path_list = sample['video']
                for vpath in visual_path_list:
                    image = Image.open(vpath).convert('RGB')
                    image = image.resize((224,224))
                    image = self.video_processor.preprocess([image],return_tensors='pt')
                    image = image['pixel_values']  # t,c,h,w
                    video.append(image)
                video = torch.cat(video,dim=0) # t,c,h,w

                masks = []
                mask_path_list = sample['mask']
                for mask_path in mask_path_list:
                    mask = cv2.imread(mask_path)
                    gray_mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
                    gt_mask = gray_mask > 0
                    gt_mask = cv2.resize(gt_mask.astype(np.float32),(224,224),interpolation=cv2.INTER_NEAREST)
                    gt_mask = torch.from_numpy(gt_mask).unsqueeze(0).to(torch.float32) # (1,224,224)
                    masks.append(gt_mask)
                masks = torch.stack(masks,dim=0) # t,1,h,w
            else:
                ## audio
                audio_path = sample['audio_path']
                i = sample['idx']
                audio, sr = librosa.load(audio_path,sr=16000,mono=True)
                length = len(audio)
                tot = 5
                nums_per_second = int(length / tot)
                indices = [i for i in range(tot)]
                audio_feature = []
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

                ## image
                vpath = sample['image_path']
                image = Image.open(vpath).convert('RGB')
                image = image.resize((224,224))
                image = self.video_processor.preprocess([image],return_tensors='pt')
                image = image['pixel_values']  # t,c,h,w
                data['image'] = image

                ## video
                image_path_list = sample['image_path_list']
                video = []
                for vpath in image_path_list:
                    image = Image.open(vpath).convert('RGB')
                    image = image.resize((224,224))
                    image = self.video_processor.preprocess([image],return_tensors='pt')
                    image = image['pixel_values']  # t,c,h,w
                    video.append(image)
                video = torch.cat(video,dim=0) # t,c,h,w
                data['video'] = video
                
                ### mask decoder
                mask_path = sample['mask_path']
                mask = cv2.imread(mask_path)
                gray_mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
                gt_mask = gray_mask > 0
                gt_mask = cv2.resize(gt_mask.astype(np.float32),(224,224),interpolation=cv2.INTER_NEAREST)
                gt_mask = torch.from_numpy(gt_mask).unsqueeze(0).to(torch.float32) # (1,224,224)
                data['mask'] = gt_mask

                ### vqgan
                # mask = cv2.imread(mask_path)
                # gray_mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
                # gt_mask = gray_mask > 0
                # gt_mask = cv2.resize(gt_mask.astype(np.float32),(224,224),interpolation=cv2.INTER_NEAREST)
                # gt_mask = torch.from_numpy(gt_mask).unsqueeze(0).to(torch.float32) # (1,224,224)
                # gt_mask = gt_mask * 255.
                # gt_mask = gt_mask.repeat(3,1,1) # 3,h,w
                # gt_mask = gt_mask / 127.5 -1

                data.update(
                    {
                        # 'audio':audio,
                        # 'image':image,
                        # 'mask':gt_mask,
                        'mask_path':mask_path,
                        'image_path':vpath,
                        'audio_path':audio_path,
                    }
                )

        elif task_name == 'avss':
            if self.multi_frames:
                pass
            else:
                audio_path = sample['audio_path']
                i = sample['idx']
                audio, sr = librosa.load(audio_path,sr=16000,mono=True)
                if len(audio) < sr: # < 1s
                    sil = np.zeros(sr-len(audio), dtype=float)
                    audio = np.concatenate((audio,sil),axis=0)
                tot = 10
                length = len(audio)
                nums_per_second = int(length / tot)
                start_time = max(0, i)
                end_time = min(tot, i + 1)
                audio = audio[int(start_time * nums_per_second) : int(end_time * nums_per_second)]
                audio = torch.from_numpy(audio) # L,
                audio = audio.unsqueeze(0)
                audio = preprocess(audio)
                audio = audio.to(torch.float32) # 1,L,128   T=1

                vpath = sample['image_path']
                image = Image.open(vpath).convert('RGB')
                image = image.resize((224,224))
                image = self.video_processor.preprocess([image],return_tensors='pt')
                image = image['pixel_values']  # t,c,h,w
                
                mask_path = sample['mask_path']
                mask = Image.open(mask_path).convert('RGB')
                mask = mask.resize((224,224),Image.Resampling.NEAREST)
                mask = color_mask_to_label(mask,self.v2_pallete) # np.array  (h,w)
                mask = torch.from_numpy(mask).unsqueeze(0).to(torch.long) # (1,224,224)

                data.update(
                    {
                        'audio':audio,
                        'image':image,
                        'mask':mask,
                        'audio_path':audio_path,
                        'image_path':vpath,
                        'mask_path':mask_path,
                    }
                )

        elif task_name == 'arig':
            audio_path = sample['audio_path']
            i = sample['idx']
            tot = sample['tot']
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)
            # if len(audio) < sr: # < 1s
            #     sil = np.zeros(sr-len(audio), dtype=float)
            #     audio = np.concatenate((audio,sil),axis=0)
            # length = len(audio)
            # nums_per_second = length / tot
            # start_time = max(0, i)
            # end_time = min(tot, i + 1)
            # audio = audio[int(nums_per_second * start_time) : int(nums_per_second * end_time)]
            length = len(audio)
            tot = 5
            nums_per_second = int(length / tot)
            indices = [i for i in range(tot)]
            audio_feature = []
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
            # data['audio'] = audio_feature

            # audio = torch.from_numpy(audio) # L,
            # audio = audio.unsqueeze(0)
            # audio = preprocess(audio)
            # audio = audio.to(torch.float32) # 1,L,128   T=1

            vpath = sample['image_path']
            image = Image.open(vpath).convert('RGB')
            image = image.resize((224,224))
            image = self.video_processor.preprocess([image],return_tensors='pt')
            image = image['pixel_values']  # t,c,h,w

            data.update(
                {
                    'audio':audio_feature,
                    'image':image,
                    'audio_path':audio_path,
                    'image_path':vpath
                }
            )

        elif task_name == 'avcap':
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
            indices = [i for i in range(10)]
            tot = 10
            nums_per_second = int(length / tot)
            for indice in indices:
                start_time = max(0, indice)
                end_time = min(tot, indice + 1)
                audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]
                # audio_seg = audio[int(indice*sr):int((indice+1)*sr)]
                # if indice - 0.5 < 0:
                #     sil = np.zeros(2 * nums_per_second - len(audio_seg), dtype=float)
                #     audio_seg = np.concatenate((sil, audio_seg),axis=0)
                if indice + 1 > tot:
                    sil = np.zeros(1 * nums_per_second - len(audio_seg), dtype=float)
                    audio_seg = np.concatenate((audio_seg, sil),axis=0)
                audio_seg = torch.from_numpy(audio_seg).unsqueeze(0)
                fbank = preprocess(audio_seg)
                fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
                audio_feature.append(fbank)
            audio_feature = torch.stack(audio_feature,dim=0) # t,L,128
            data['audio'] = audio_feature
            data['audio_path'] = audio_path

        elif task_name == 'ref-avs':
            ## video
            image_path_list = sample['image_path_list']
            video = []
            for path in image_path_list:
                image = Image.open(path).convert('RGB')
                image = image.resize((224,224))
                image = self.video_processor.preprocess([image],return_tensors='pt')
                image = image['pixel_values']  # t,c,h,w
                video.append(image)
            video = torch.cat(video,dim=0)
            data['video'] = video

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
class DataCollatorForUnifiedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        
        tokenizer=self.tokenizer
        batch_input_ids=[]
        batch_label=[]
        batch_X_modals=[]
        batch_task_names = []

        for instance in instances:
            instruction=instance['instruction']
            output=instance['output']
            task_name = instance['task_name']
            batch_task_names.append(task_name)
            
            instruction_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(instruction))
            output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))
            input_ids = instruction_ids + output_ids
            label = [-100] * len(instruction_ids) + output_ids
            batch_input_ids.append(torch.tensor(input_ids,dtype=torch.long))
            batch_label.append(torch.tensor(label,dtype=torch.long))
            
            X_modals = {}
            image = instance.get('image',None)
            if image is not None:
                X_modals['<image>'] = image
                
            video = instance.get('video',None)
            if video is not None:
                X_modals['<video>'] = video

            audio = instance.get('audio',None)
            if audio is not None:
                X_modals['<audio>'] = audio
            
            mask = instance.get('mask',None)
            if mask is not None:
                X_modals['<mask>'] = mask
            
            batch_X_modals.append(X_modals)

        
        return {
            'batch_input_ids':batch_input_ids,
            'batch_labels':batch_label,
            'batch_X_modals':batch_X_modals,
            'batch_task_names':batch_task_names
        }


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
            
            if task_name == 'avqa':
                question_type = instance.get('question_type',None)
                vid = instance.get('vid',None)
                qid = instance.get('qid',None)
                metadata.update(
                    {
                        'question_type':question_type,
                        'vid':vid,
                        'qid':qid
                    }
                )
            
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
    image_processor=None,mode='train',
    image_scale_nums = 2, token_nums_per_scale = 3, test_name = 'test_s',
):
    if mode == 'train':
        dataset = UnifiedDataset(
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
            avcap_task=data_args.avcap_task,
            multi_frames=data_args.multi_frames,
            image_scale_nums=image_scale_nums,
            token_nums_per_scale=token_nums_per_scale
        )
        data_collator = DataCollatorForUnifiedDataset(tokenizer=tokenizer)
    
    elif mode == 'test':
        dataset = UnifiedTestDataset(
            video_processor=image_processor,
            tokenizer=tokenizer,
            avqa_task=data_args.avqa_task,
            ave_task=data_args.ave_task,
            avvp_task = data_args.avvp_task,
            arig_task = data_args.arig_task, 
            avcap_task = data_args.avcap_task,
            avss_task=data_args.avss_task,
            ms3_task=data_args.ms3_task,
            s4_task=data_args.s4_task,
            ref_avs_task=data_args.ref_avs_task,
            test_name=test_name,
            multi_frames=data_args.multi_frames,
            image_scale_nums=image_scale_nums,
            token_nums_per_scale=token_nums_per_scale
        )
        data_collator = DataCollatorForUnifiedTestDataset(tokenizer=tokenizer)
    
    return dataset,data_collator


