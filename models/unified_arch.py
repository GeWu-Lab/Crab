import torch
from abc import ABC, abstractmethod
from torch import nn

from models.multimodal_encoder import (
    VisualEncoder,
    AudioEncoder,
    VLProjector,
    ALProjector,
    SegModule,
    MaskEncoder
)

class UnifiedMetaModel:

    def __init__(self, config):
        super(UnifiedMetaModel, self).__init__(config)
        self.config = config


    def init_multimodal_modules(
        self,
        d_model = 3584,
        # visual
        vit_ckpt_path = '/group/40061/cserdu/pretrain/openai-clip-vit-large-patch14-224',
        select_layer_list = [-11,-1],
        select_feature = 'patch',
        image_size = 224,
        patch_size = 14,
        visual_query_token_nums = 32,
        # audio
        BEATs_ckpt_path = '/group/40061/cserdu/pretrain/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt',
        audio_query_token_nums = 32,
        # seg
        image_scale_nums = 2,
        token_nums_per_scale = 3,
        avs_query_num = 300,
        num_classes = 1,
        query_generator_num_layers = 2,
        prompt_embed_dim = 256,
        mask_decoder_transformer_depth = 2,
        low_res_mask_size = 112,
        dice_loss_weight = 0.5,
        bce_loss_weight = 2.0,
        vit_image_embedding_dim = 1024,
        visual_branch = False,
        audio_branch = False,
        segment_branch = False,
        # vqgan
        use_vqgan = False,
    ):
        
        if visual_branch:
            image_token_nums = (image_size//patch_size) * (image_size//patch_size)
            self.visual_encoder = VisualEncoder(model_name_or_path=vit_ckpt_path,select_layer_list=select_layer_list,
                                                select_feature=select_feature)
            self.vl_projector = VLProjector(hidden_size=1024, d_model=d_model, depth=2, image_token_nums=image_token_nums,
                                            num_query_token=visual_query_token_nums, num_hidden_layers=2,)
            print('init visual_encoder, vl_projector finished...')

        if audio_branch:
            self.audio_encoder =  AudioEncoder(ckpt_path=BEATs_ckpt_path)
            self.al_projector = ALProjector(hidden_size=768, d_model=d_model, depth=2, num_query_token=audio_query_token_nums,
                                            num_hidden_layers=2)
            print('init audio_encoder, al_projector finished...')


        if segment_branch:
            self.low_res_mask_size = low_res_mask_size
            self.seg_module = SegModule(
                d_model=d_model,
                prompt_embed_dim=prompt_embed_dim,
                image_scale_nums=image_scale_nums,
                token_nums_per_scale=token_nums_per_scale,
                mask_decoder_transformer_depth=mask_decoder_transformer_depth,
                vit_image_embedding_dim=vit_image_embedding_dim,
                avs_query_num=avs_query_num,
                num_classes=num_classes,
                query_generator_num_layers=query_generator_num_layers,
                image_size=image_size,
                patch_size=patch_size,
                image_embedding_size=(image_size // patch_size),
                dice_loss_weight=dice_loss_weight,
                bce_loss_weight=bce_loss_weight,
            )
            print('init seg_module finished...')

        if use_vqgan:
            self.mask_encoder = MaskEncoder(token_shift=32000+20)


    def encode_video(self,visual,batch_question=None):
        vit_feature_list = self.visual_encoder(visual)  # [(b,t*n,d),(b,t*n,d),...]
        qformer_feature_list = []
        for vit_feature in vit_feature_list:
            qformer_feature = self.vl_projector(vit_feature,batch_question) # b,t*256 -> b,t*32
            qformer_feature_list.append(qformer_feature)
        return vit_feature_list,qformer_feature_list


    def encode_audio(self,audio,batch_qustion=None):
        audio_feature = self.audio_encoder(audio)
        audio_feature = self.al_projector(audio_feature,batch_qustion)
        return audio_feature


    def encode_mask(self,mask):
        return self.mask_encoder(mask)


    def postprocess_seg(
        self,
        pred_embeddings, # bs,nums,dim
        multi_scale_image_feature_list,
        gt_mask = None,
        batch_task_names = [],
    ):
        low_res_mask_size = self.low_res_mask_size
        return self.seg_module(
            pred_embeddings = pred_embeddings,
            multi_scale_image_feature_list = multi_scale_image_feature_list,
            low_res_mask_size = low_res_mask_size,
            gt_mask = gt_mask,
            batch_task_names = batch_task_names,
        )


class UnifiedMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self) -> UnifiedMetaModel:
        pass

    def encode_audio(self,audio,batch_qustion=None, batch_first=True):
        if not batch_first:
            audio = audio.unsqueeze(0)
        audio_feature = self.get_model().encode_audio(audio,batch_qustion=batch_qustion)
        if not batch_first:
            audio_feature = audio_feature.squeeze(0)
        return audio_feature


    def encode_video(self,video,batch_question=None,batch_first=True):
        if not batch_first:
            video = video.unsqueeze(0)
        vit_feature_list, qformer_feature_list = self.get_model().encode_video(video,batch_question=batch_question)
        if not batch_first:
            vit_feature_list = [item.squeeze(0) for item in vit_feature_list]
            qformer_feature_list = [item.squeeze(0) for item in qformer_feature_list]
        return vit_feature_list,qformer_feature_list


    def encode_mask(self,mask,batch_first=False):
        if not batch_first:
            mask = mask.unsqueeze(0)
        indices = self.get_model().encode_mask(mask) # b,n
        if not batch_first:
            indices = indices.squeeze(0)
        return indices
    

    def encode_ids(self,ids):
        return self.get_model().embed_tokens(ids)
    
    
    def prepare_multimodal_inputs(
        self,
        batch_input_ids,
        batch_labels,
        batch_X_modals,
        batch_audio_question = None,
        batch_visual_question = None,
        batch_task_names = None,
        return_multi_scale_features = False,
        return_gt_mask = False,
    ):
        device = self.device
        bs = len(batch_input_ids)

        gt_mask = []
        scale = 2
        multi_scale_image_features = [[] for _ in range(scale)]
        if return_gt_mask:
            for i, X_modals in enumerate(batch_X_modals):
                is_avs_task = batch_task_names[i] in ['ms3','s4','avss','ref-avs']
                if not is_avs_task:
                    continue
                for key, X_modal in X_modals.items():
                    if key == '<mask>':
                        gt_mask.append(X_modal) # [(1,224,224), ...]
                    elif key == '<image>':
                        vit_feature_list, qformer_feature_list = self.encode_video(batch_X_modals[i][key],batch_question=None,batch_first=False)
                        if return_multi_scale_features and is_avs_task:
                            for _scale in range(scale):
                                multi_scale_image_features[_scale].append(vit_feature_list[_scale])
            if len(gt_mask) > 0:
                gt_mask = torch.stack(gt_mask,dim=0) # b,1,224,224

        max_length = 0
        new_batch_inputs_embeds = []
        new_batch_attention_mask = []
        new_batch_labels = []
        batch_mask_token_indices = []
        # keys = ['<image>','<video>','<audio>','<mask>']
        keys = self.KEYS
        mask_tokens = self.MASK
        for i in range(bs):
            input_ids = batch_input_ids[i]
            labels = batch_labels[i]
            task_name = batch_task_names[i]
            is_avs_task = task_name in ['ms3','s4','avss','ref-avs']

            if return_multi_scale_features and is_avs_task:
                mask_token_indices = torch.where(torch.any(torch.stack([input_ids == self.SPECIAL_TOKEN_2_IDS[mask_token] for mask_token in mask_tokens]), dim=0))[0]
                mask_token_indices = mask_token_indices.tolist()
            
            X_token_indices = torch.where(torch.any(torch.stack([input_ids == self.SPECIAL_TOKEN_2_IDS[key] for key in keys]), dim=0))[0]
            X_token_indices = X_token_indices.tolist()            

            inputs_embeds_seg=[]
            labels_seg=[]
            pre_indice=0
            for idx,indice in enumerate(X_token_indices):
                inputs_embeds_seg.append(self.encode_ids(input_ids[pre_indice:indice]))
                labels_seg.append(labels[pre_indice:indice])
                
                special_token = self.IDS_2_SPECIAL_TOKEN[input_ids[indice].item()]
                if special_token == '<audio>':
                    feature = self.encode_audio(batch_X_modals[i][special_token],batch_qustion=None,batch_first=False)
                    inputs_embeds_seg.append(feature)
                    labels_seg.append(torch.full((feature.shape[0],),-100,dtype=torch.long,device=device))
                elif special_token == '<video>':
                    vit_feature_list, qformer_feature_list = self.encode_video(batch_X_modals[i][special_token],batch_question=None,batch_first=False)
                    feature = qformer_feature_list[-1] # last layer qformer feature
                    inputs_embeds_seg.append(feature)  
                    labels_seg.append(torch.full((feature.shape[0],),-100,dtype=torch.long,device=device))
                    # if return_multi_scale_features and is_avs_task: # for ref-avs
                    #     for _scale in range(scale):
                    #         multi_scale_image_features[_scale].append(vit_feature_list[_scale]) # (10*256,1024)
                elif special_token == '<image>':
                    vit_feature_list, qformer_feature_list = self.encode_video(batch_X_modals[i][special_token],batch_question=None,batch_first=False)
                    feature = qformer_feature_list[-1] # last layer qformer feature
                    inputs_embeds_seg.append(feature)
                    labels_seg.append(torch.full((feature.shape[0],),-100,dtype=torch.long,device=device))
                    # if return_multi_scale_features and is_avs_task:
                    #     for _scale in range(scale):
                    #         multi_scale_image_features[_scale].append(vit_feature_list[_scale])
                elif special_token == '<mask>':  # for vqgan
                    indices = self.encode_mask(batch_X_modals[i][special_token],batch_first=False)
                    feature = self.encode_ids(indices)
                    inputs_embeds_seg.append(feature)
                    labels_seg.append(indices)
                
                if return_multi_scale_features and is_avs_task:
                    mask_token_indices = [item + feature.shape[0] - 1 for item in mask_token_indices]

                pre_indice = indice + 1 # +1,skip special token

            # add last tokens
            inputs_embeds_seg.append(self.encode_ids(input_ids[pre_indice:]))
            labels_seg.append(labels[pre_indice:])
            # concat segs
            inputs_embeds_seg = torch.cat(inputs_embeds_seg,dim=0)
            attention_mask_seg = torch.ones(inputs_embeds_seg.shape[0],dtype=torch.int32,device=device)
            labels_seg = torch.cat(labels_seg,dim=0)
            
            new_batch_inputs_embeds.append(inputs_embeds_seg)
            new_batch_attention_mask.append(attention_mask_seg)
            new_batch_labels.append(labels_seg)
            
            if return_multi_scale_features and is_avs_task:
                batch_mask_token_indices.append(mask_token_indices)
            # batch_mask_token_indices.append(mask_token_indices)

            max_length = max(max_length,inputs_embeds_seg.shape[0])


        if return_multi_scale_features:
            ## [(bs,n,dim), (bs,n,dim), ...]
            multi_scale_image_features = [torch.stack(item,dim=0) for item in multi_scale_image_features if len(item) > 0]

        ### left padding
        padding_inputs_embeds = []
        padding_attention_mask = []
        padding_labels = []
        padding_mask_token_mask = []
        avs_sample_idx = 0
        for i in range(bs):
        # for embeds, mask, labels, mask_indice in zip(new_batch_inputs_embeds, new_batch_attention_mask, new_batch_labels, batch_mask_token_indices):
            embeds = new_batch_inputs_embeds[i]
            mask = new_batch_attention_mask[i]
            labels = new_batch_labels[i]
            
            L,d = embeds.shape
            pad_embeds = self.encode_ids(torch.full((max_length-L,),self.get_model().pad_token_id,dtype=torch.long,device=device))
            padding_inputs_embeds.append(torch.cat([pad_embeds,embeds],dim=0))
            padding_attention_mask.append(torch.cat([torch.zeros((max_length-L),dtype=torch.int32,device=device),mask],dim=0))
            padding_labels.append(torch.cat([torch.full((max_length-L,),-100,dtype=torch.long,device=device),labels],dim=0))
            
            task_name = batch_task_names[i]
            is_avs_task = task_name in ['ms3','s4','avss','ref-avs']
            if return_multi_scale_features and is_avs_task:
                mask_indice = batch_mask_token_indices[avs_sample_idx]
                mask_indice = [item + max_length - L - 1 for item in mask_indice]  # note: -1
                mask_token_mask = torch.zeros((max_length),dtype=torch.bool,device=device)
                mask_token_mask[mask_indice] = 1
                padding_mask_token_mask.append(mask_token_mask)
                avs_sample_idx += 1
    
        padding_inputs_embeds = torch.stack(padding_inputs_embeds,dim=0)
        padding_attention_mask = torch.stack(padding_attention_mask,dim=0)
        padding_labels = torch.stack(padding_labels,dim=0)
        if len(padding_mask_token_mask) > 0:
            padding_mask_token_mask = torch.stack(padding_mask_token_mask,dim=0)

        position_ids = torch.cumsum(padding_attention_mask,dim=-1) - 1
        position_ids[position_ids==-1] = 0

        # if has_avs_sample:
        #     print(batch_task_names)
        #     print(f'prepare: padding mask token: {padding_mask_token_mask.shape} gt_mask: {gt_mask.shape} len(multi_scale): {len(multi_scale_image_features)} {multi_scale_image_features[0].shape}')

        ### right padding ###
        # padding_inputs_embeds=[]
        # for emb in new_batch_inputs_embeds:
        #     L,d=emb.shape
        #     pad_embeds = self.get_model().embed_tokens(torch.full(max_length-L,self.pad_token_id,dtype=torch.long,device=device))
        #     padding_inputs_embeds.append(torch.cat([emb,pad_embeds],dim=0))
        # padding_inputs_embeds=torch.stack(padding_inputs_embeds,dim=0)

        # padding_attention_mask = pad_sequence(new_batch_attention_mask,batch_first=True,padding_value=0)
        # padding_labels = pad_sequence(new_batch_labels,batch_first=True,padding_value=-100)

        # position_ids = torch.cumsum(padding_attention_mask,dim=-1) - 1
        # position_ids[position_ids==-1] = 0
        
        dict_data = {
            'input_ids':None,
            'inputs_embeds':padding_inputs_embeds,
            'attention_mask':padding_attention_mask,
            'labels':padding_labels,
            'position_ids':position_ids,
        }
        if return_multi_scale_features:
            dict_data['multi_scale_image_features'] = multi_scale_image_features
            dict_data['mask_token_mask'] = padding_mask_token_mask
        if return_gt_mask:
            dict_data['gt_mask'] = gt_mask
        
        return dict_data
    

    def initialize_MM_tokenizer(self, tokenizer, mask_token_nums = 6, output_embeddings_require_grad = False, use_vqgan = False):
        vocab_nums=len(tokenizer)
        added_tokens = []
        image_tokens = ['<image>','<image_start>','<image_end>']
        added_tokens += image_tokens
        video_tokens = ['<video>','<video_start>','<video_end>']
        added_tokens += video_tokens
        audio_tokens = ['<audio>','<audio_start>','<audio_end>']
        added_tokens += audio_tokens
        mask_tokens = ['<mask_start>','<mask_end>']
        added_tokens += mask_tokens
        num_new_tokens = tokenizer.add_tokens(added_tokens,special_tokens=True)

        if use_vqgan:
            vqgan_vocab_nums = 16384
            vqgan_tokens = ['<mask>','<vqgan_start>','<vqgan_end>'] + [f'<vqgan_{i}>' for i in range(vqgan_vocab_nums)]
            added_tokens += vqgan_tokens
            self.KEYS.append('<mask>')

        seg_tokens = [f'<mask_{i}>' for i in range(mask_token_nums)]
        num_new_tokens += tokenizer.add_tokens(seg_tokens,special_tokens=True)
        added_tokens += seg_tokens
        
        ### tag token
        # tag_tokens = ['<event>','</event>','<range>','</range>','<visual_event>',
        #               '</visual_event>','<audio_event>','</audio_event>','<obj>','</obj>',
        #               '<answer>','</answer>']
        # num_new_tokens += tokenizer.add_tokens(tag_tokens,special_tokens=True)
        # added_tokens += tag_tokens

        # loc_tokens = [f'<loc_{i}>' for i in range(224)]
        # num_new_tokens += tokenizer.add_tokens(loc_tokens,special_tokens=False)
        # added_tokens += loc_tokens

        self.KEYS = ['<image>','<video>','<audio>']
        self.MASK = seg_tokens
        
        self.SPECIAL_TOKEN_2_IDS={
            token : i + vocab_nums for i,token in enumerate(added_tokens)
        }
        self.IDS_2_SPECIAL_TOKEN={
            i + vocab_nums:token for i,token in enumerate(added_tokens)
        }
        # print(self.IDS_2_SPECIAL_TOKEN)
        '''
            {'<image>': 151646, '<image_start>': 151647, '<image_end>': 151648, '<video>': 151649, '<video_start>': 151650, 
            '<video_end>': 151651, '<audio>': 151652, '<audio_start>': 151653, '<audio_end>': 151654, '<mask_start>': 151655, 
            '<mask_end>': 151656, '<mask_0>': 151657, '<mask_1>': 151658, '<mask_2>': 151659, '<mask_3>': 151660, 
            '<mask_4>': 151661, '<mask_5>': 151662}
        '''
        self.resize_token_embeddings(len(tokenizer))
        # self.get_input_embeddings().requires_grad_(True)

        # if num_new_tokens > 0:
        #     input_embeddings = self.get_input_embeddings().weight.data
        #     output_embeddings = self.get_output_embeddings().weight.data

        #     input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
        #         dim=0, keepdim=True)
        #     output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
        #         dim=0, keepdim=True)

        #     input_embeddings[-num_new_tokens:] = input_embeddings_avg
        #     output_embeddings[-num_new_tokens:] = output_embeddings_avg

        #     self.get_input_embeddings().requires_grad_(True)
        #     if output_embeddings_require_grad:
        #         self.get_output_embeddings().requires_grad_(True)
        #     else:
        #         self.get_output_embeddings().requires_grad_(False)

        # self.get_input_embeddings().requires_grad_(False)
        # if output_embeddings_require_grad:
        #     self.get_output_embeddings().requires_grad_(True)
        # else:
        #     self.get_output_embeddings().requires_grad_(False)


    @property
    def device(self):
        return list(self.parameters())[0].device


