import json
import ast
import os
from os.path import join
import numpy as np
from typing import Sequence,Dict
from dataclasses import dataclass

import torch
import transformers
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils.mm_utils import process_video

'''
flute', 'bagpipe', 'guzheng', 'more than ten', 'cello', 'tuba', 'violin', 'seven', 'five', 'congas', 'two', \
'ukulele', 'four', 'right', 'ten', 'eight', 'simultaneously', 'yes', 'six', 'drum', 'no', 'suona', 'electric_bass', \
'zero', 'trumpet', 'banjo', 'piano', 'saxophone', 'middle', 'left', 'nine', 'acoustic_guitar', 'bassoon', 'pipa', \
'three', 'outdoor', 'accordion', 'xylophone', 'one', 'clarinet', 'erhu', 'indoor'
'''
'''
train: 31927
test: 9129
'''

class AVQADataset(Dataset):

    def __init__(
        self,
        annotation_path,
        video_processor,
        image_aspect_ratio='pad',
        data_root='/data/users/guangyao_li/MUSIC-AVQA',
        num_frames=8,
    ) -> None:
        super().__init__()

        self.video_processor=video_processor
        self.aspect_ratio=image_aspect_ratio
        self.num_frames=num_frames
        samples=json.load(open(annotation_path,'r'))
        
        data=[]

        for sample in samples:
            video_id=sample['video_id']
            question_id=sample['question_id']
            type=ast.literal_eval(sample['type'])
            question_content=sample['question_content']
            if '\uff1f' in question_content:
                question_content=question_content.replace('\uff1f','?')
            templ_values=ast.literal_eval(sample['templ_values'])
            answer=sample['anser']

            idx=0
            words=question_content.split(' ')
            for i,word in enumerate(words):
                if '<' in word:
                    words[i]=templ_values[idx]
                    idx+=1
            
            question=' '.join(words)
            if question[-1]!='?':
                question=question+'?'

            data.append(
                {
                    'video_id':video_id,
                    'question_id':question_id,
                    'type':type,
                    'video_path':join(data_root,'avqa-videos',str(video_id)+'.mp4'),
                    'audio_path':join(data_root,'vggish',str(video_id)+'.npy'),
                    'question':question,
                    'answer':answer
                }
            )

        self.data=data

    
    def __len__(self):
        return len(self.data)


    def __getitem__(self,idx):
        sample=self.data[idx]
        video_path=sample['video_path']
        audio_path=sample['audio_path']
        video_id=sample['video_id']
        question_id=sample['question_id']
        type=sample['type']

        video=process_video(video_path,self.video_processor,aspect_ratio=self.aspect_ratio,num_frames=self.num_frames,sample_scheme='uniform')
        audio=np.load(audio_path)
        idx=np.linspace(0, 59, self.num_frames, dtype=int)
        idx=np.array(idx)
        audio=torch.tensor(audio[idx],dtype=torch.float32)

        question=sample['question']
        instruction=f"[INST] Based on the video <video> and audio <audio>, please answer the question: {question} [/INST]"
        
        answer=sample['answer']
        output=f'According to the video and audio, the answer is {answer}.'

        return {
            'instruction':instruction,
            'output':output,
            'video':video,
            'audio':audio,
            'video_id':video_id,
            'question_id':question_id,
            'type':type
        }


@dataclass
class DataCollatorForAVQADataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        
        tokenizer=self.tokenizer

        batch_input_ids=[]
        batch_label=[]
        batch_mask=[]
        
        batch_video=[]
        batch_audio=[]
        batch_video_id=[]
        batch_question_id=[]
        batch_type=[]
        batch_output=[]

        for instance in instances:

            instruction=instance['instruction']
            output=instance['output']
            video=instance['video']
            audio=instance['audio']
            video_id=instance['video_id']
            question_id=instance['question_id']
            type=instance['type']

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
            batch_question_id.append(question_id)
            batch_video_id.append(video_id)
            batch_type.append(type)
            batch_output.append(output)

        batch_input_ids = pad_sequence(batch_input_ids,batch_first=True,padding_value=tokenizer.pad_token_id)
        batch_label = pad_sequence(batch_label,batch_first=True,padding_value=-100)
        batch_mask = pad_sequence(batch_mask,batch_first=True,padding_value=0)
        batch_video = torch.stack(batch_video,dim=0) # b,t,c,h,w
        batch_audio = torch.stack(batch_audio,dim=0) # b,t,d
        
        return {
            'input_ids':batch_input_ids,
            'labels':batch_label,
            'attention_mask':batch_mask,
            'video':batch_video,
            'audio':batch_audio,
            'batch_video_id':batch_video_id,
            'batch_question_id':batch_question_id,
            'batch_type':batch_type,
            'batch_output':batch_output
        }



@dataclass
class DataCollatorForAVQATestDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        
        tokenizer=self.tokenizer

        batch_input_ids=[]
        batch_label=[]
        batch_mask=[]
        
        batch_video=[]
        batch_audio=[]
        batch_video_id=[]
        batch_question_id=[]
        batch_type=[]
        batch_output=[]

        for instance in instances:

            instruction=instance['instruction']
            output=instance['output']
            video=instance['video']
            audio=instance['audio']
            video_id=instance['video_id']
            question_id=instance['question_id']
            type=instance['type']

            instruction_ids = [tokenizer.bos_token_id]+tokenizer.convert_tokens_to_ids(tokenizer.tokenize(instruction))
            # output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))+[tokenizer.eos_token_id]
            
            input_ids=instruction_ids
            label=[-100]*len(instruction_ids)
            mask=[1]*len(input_ids)
        
            batch_input_ids.append(torch.tensor(input_ids,dtype=torch.long))
            batch_label.append(torch.tensor(label,dtype=torch.long))
            batch_mask.append(torch.tensor(mask,dtype=torch.int32))
            
            batch_audio.append(audio)
            batch_video.append(video)
            batch_question_id.append(question_id)
            batch_video_id.append(video_id)
            batch_type.append(type)
            batch_output.append(output)

        batch_input_ids = pad_sequence(batch_input_ids,batch_first=True,padding_value=tokenizer.pad_token_id)
        batch_label = pad_sequence(batch_label,batch_first=True,padding_value=-100)
        batch_mask = pad_sequence(batch_mask,batch_first=True,padding_value=0)
        batch_video = torch.stack(batch_video,dim=0) # b,t,c,h,w
        batch_audio = torch.stack(batch_audio,dim=0) # b,t,d
        
        return {
            'input_ids':batch_input_ids,
            'labels':batch_label,
            'attention_mask':batch_mask,
            'video':batch_video,
            'audio':batch_audio,
            'batch_video_id':batch_video_id,
            'batch_question_id':batch_question_id,
            'batch_type':batch_type,
            'batch_output':batch_output
        }


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = AVQADataset(
        annotation_path=data_args.annotation_path,
        video_processor=data_args.video_processor,
        image_aspect_ratio=data_args.image_aspect_ratio,
        data_root=data_args.data_root,
        num_frames=data_args.num_frames,
    )
    data_collator = DataCollatorForAVQADataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


