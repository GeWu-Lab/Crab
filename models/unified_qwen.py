import json
import torch
from torch import nn
from typing import Optional,List,Tuple
from transformers import Qwen2ForCausalLM,Qwen2Model,AutoModelForCausalLM,Qwen2Config,AutoConfig

from models.unified_arch import UnifiedMetaModel,UnifiedMetaForCausalLM

class UnifiedConfig(Qwen2Config):
    model_type = "unified_llm"


class UnifiedModel(UnifiedMetaModel,Qwen2Model):
    config_class = UnifiedConfig

    def __init__(self, config: Qwen2Config, **kwargs):
        super(UnifiedModel, self).__init__(config, **kwargs)
        self.config = config


class UnifiedForCausalLM(Qwen2ForCausalLM,UnifiedMetaForCausalLM):
    config_class = UnifiedConfig

    def __init__(self, config: Qwen2Config, **kwargs):
        super().__init__(config)
        self.config=config
        self.model = UnifiedModel(config,**kwargs)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def get_model(self) -> UnifiedModel:
        return self.model


    def forward(
        self,
        batch_input_ids = None,
        batch_labels = None,
        batch_attention_mask = None,
        batch_X_modals = None,
        batch_question = None,
        # used for inference
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):

        if input_ids is not None and input_ids.shape[1]==1:
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            input_ids = None

        elif inputs_embeds is None and batch_input_ids is not None:
            inputs = self.prepare_multimodal_inputs(
                batch_input_ids=batch_input_ids,
                batch_attenion_mask=batch_attention_mask,
                batch_labels=batch_labels,
                batch_X_modals=batch_X_modals,
                batch_question=batch_question,
                return_multi_scale_features=False,
                return_gt_mask=False,
            )

            input_ids = inputs['input_ids']
            inputs_embeds = inputs['inputs_embeds']
            attention_mask = inputs['attention_mask']
            labels = inputs['labels']
            position_ids = inputs['position_ids']
            mask_token_mask = inputs['mask_token_mask']
            multi_scale_image_features = inputs.get('multi_scale_image_features',None)
            gt_mask = inputs.get('gt_mask',None)

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        # if '<mask>' in list(batch_X_modals[0].keys()):
        #     output_hidden_states = output.hidden_states
        #     # print('output_hidden_states len: ',len(output_hidden_states)) # 29
        #     # print('output_hidden_states[-1]: ',output_hidden_states[-1].shape) # torch.Size([2, 350, 3584])
        #     bs,_,dim = output_hidden_states[-1].shape
        #     pred_embeddings = output_hidden_states[-1][mask_token_mask] # L,dim
        #     pred_embeddings = pred_embeddings.reshape(bs,-1,dim) # bs,n,dim

        #     seg_output = self.model.postprocess_seg(
        #         pred_embeddings=pred_embeddings,
        #         multi_scale_image_feature_list=multi_scale_image_features,
        #         gt_mask=gt_mask,
        #         low_res_mask_size=112
        #     )
        #     mask_loss = seg_output['mask_loss']
        #     loss = output.loss * 1.0 + mask_loss
        #     output.loss = loss
        #     return output
        # else:
        #     return output

        ## inference segmentation
        return output


    @torch.no_grad()
    def generate(
        self,
        batch_input_ids,
        batch_attention_mask,
        batch_labels,
        batch_X_modals,
        **kwargs
    ):
        inputs = self.prepare_multimodal_inputs(
            batch_input_ids=batch_input_ids,
            batch_attenion_mask=batch_attention_mask,
            batch_labels=batch_labels,
            batch_X_modals=batch_X_modals,
        )
        input_ids = inputs['input_ids']
        inputs_embeds = inputs['inputs_embeds']
        # print('inputs_embeds.shape: ',inputs_embeds.shape)
        attention_mask = inputs['attention_mask']
        labels = inputs['labels']
        position_ids = inputs['position_ids']
        mask_token_mask = inputs['mask_token_mask']
        # print('mask_token_mask.shape: ',mask_token_mask.shape)

        multi_scale_image_features = inputs['multi_scale_image_features']
        gt_mask = inputs['gt_mask']

        output = super().generate(
            # position_ids = position_ids,
            # attention_mask = attention_mask,
            inputs_embeds = inputs_embeds,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **kwargs
        )
        # print('output.hiden_states: ',len(output.hidden_states))
        
        output_hidden_states = output.hidden_states
        # # print('type(output_hidden_states): ',type(output_hidden_states))
        # print('len(output_hidden_states): ',len(output_hidden_states),' output_hidden_states[-1]: ',output_hidden_states[-1].shape)
        output_ids = output.sequences
        # print('output_ids: ',output_ids)
        # Sseg_token_num = self.token_nums_per_scale

        seg_token_mask = torch.zeros_like(output_ids[:, 1:]).bool()
        special_tokens = [f'<mask_{i}>' for i in range(6)]
        seg_token_idx = [self.SPECIAL_TOKEN_2_IDS[special_token] for special_token in special_tokens]
        # print('seg_token_idx: ',seg_token_idx)
        for idx in seg_token_idx:
            seg_token_mask = seg_token_mask | (output_ids[:, 1:] == idx)
        # print('seg_token_mask: ',seg_token_mask)
        mask_list =  seg_token_mask.int().tolist()[0]  # bs == 1
        # print('mask_list: ',mask_list)
        pred_embeddings = []
        for item, hs in zip(mask_list,output_hidden_states):
            if item == 1:
                # print('hs: ',hs[-1].shape)
                pred_embeddings.append(hs[-1])
        
        if len(pred_embeddings) == 0:
            return None
        
        pred_embeddings = torch.cat(pred_embeddings,dim=1)
        # print('pred_mbedings: ', pred_embeddings.shape)  # bs,n,dim

        if pred_embeddings.shape[1] >= 6:
            print(f'pred_embeddings.shape[1]>=6, shape: {pred_embeddings.shape}')
            pred_embeddings = pred_embeddings[:,-6:]
        else:
            return None

        
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        # seg_token_mask = torch.cat(
        #     [
        #         mask_token_mask,
        #         seg_token_mask,
        #     ],
        #     dim=1,
        # )

        result = self.model.postprocess_seg(
            pred_embeddings=pred_embeddings,
            multi_scale_image_feature_list=multi_scale_image_features,
            gt_mask=None
        )
        result['output_ids'] = output_ids

        return result
    

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # print('into prepare inputs...   input_ids:  ',input_ids,'  past key values:  ',past_key_values is None, '   inputs_emebds: ',inputs_embeds is None)
        # if inputs_embeds is not None:
        #     print(inputs_embeds.shape)
        # if past_key_values is not None:
        #     print(f'past key values:  {past_key_values[10][0].shape}')
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        # print(f'_inputs>>>>>  {_inputs.keys()}')
        # if 'input_ids' in _inputs.keys():
        #     print(_inputs['input_ids'])
        # if 'inputs_embeds' in _inputs.keys():
        #     print(_inputs['inputs_embeds'].shape)
        if images is not None:
            _inputs['images'] = images
        return _inputs

    
    @property
    def device(self):
        return list(self.parameters())[0].device
    

AutoConfig.register("unified_llm", UnifiedConfig)
AutoModelForCausalLM.register(UnifiedConfig, UnifiedForCausalLM)


