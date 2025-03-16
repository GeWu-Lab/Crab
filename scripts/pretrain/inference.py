import os,sys
sys.path.append(os.getcwd())
from os.path import join
import numpy as np
import pathlib
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataclasses import asdict
from PIL import Image
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    print('no npu!')

import transformers

from configs.unified_config import ModelArguments,DataArguments,TrainingArguments
from scripts.pretrain.trainer import UnifiedTrainer

from dataset.pretrain_dataset import get_dataset_collator
from utils.util import set_seed,rank0_print,find_all_linear_names,write2json,prepare_sample
from utils.deepspeed_utils import *
from utils.avss_utils import mask_iou

local_rank = None

def train(attn_implementation=None):
    global local_rank
    set_seed(42)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.llm_name == 'llama':
        d_model = 4096
    elif model_args.llm_name == 'qwen':
        d_model = 3584

    local_rank = training_args.local_rank
    compute_dtype = torch.float32
    if training_args.fp16:
        compute_dtype = torch.float16
    elif training_args.bf16:
        compute_dtype = torch.bfloat16
    
    pretrain_model_name_or_path = model_args.model_name_or_path
    if model_args.llm_name == 'llama':
        from models.unified_llama import UnifiedForCausalLM
        from transformers import LlamaConfig
        config = LlamaConfig.from_pretrained(pretrain_model_name_or_path, local_files_only=True)
        config._attn_implementation = attn_implementation
        model = UnifiedForCausalLM.from_pretrained(
            pretrain_model_name_or_path,
            config=config,
            torch_dtype=compute_dtype
        )
    elif model_args.llm_name == 'qwen':
        from models.unified_qwen import UnifiedForCausalLM
        from transformers import Qwen2Config
        config = Qwen2Config.from_pretrained(pretrain_model_name_or_path, local_files_only=True)
        config._attn_implementation = attn_implementation
        model = UnifiedForCausalLM.from_pretrained(
            pretrain_model_name_or_path,
            config = config,
            torch_dtype = compute_dtype
        )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        # rank0_print(local_rank, "Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        print('Add lora adapters finished...')

    if model_args.llm_name == 'qwen':
        from transformers import Qwen2Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(
            pretrain_model_name_or_path,
            padding_side="left",
            use_fast=True,
        )
    
    elif model_args.llm_name == 'llama':
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            pretrain_model_name_or_path,
            padding_side="left",
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    ori_tokenizer_vocab_nums = len(tokenizer)
    model.get_model().pad_token_id = tokenizer.pad_token_id

    image_scale_nums = model_args.image_scale_nums
    token_nums_per_scale = model_args.token_nums_per_scale
    output_embeddings_require_grad = True if training_args.seg_branch else False
    model.initialize_MM_tokenizer(tokenizer,mask_token_nums=image_scale_nums*token_nums_per_scale,
                                  output_embeddings_require_grad=output_embeddings_require_grad)
    MM_tokenizer_vocab_nums = len(tokenizer)
    print('ori_tokenizer_vocab_nums: ',ori_tokenizer_vocab_nums,' MM_tokenizer_vocab_nums: ',MM_tokenizer_vocab_nums)

    model.get_model().init_multimodal_modules(visual_branch=training_args.visual_branch,
                                              audio_branch=training_args.audio_branch,
                                              segment_branch=training_args.seg_branch,
                                              d_model=d_model,vit_ckpt_path=model_args.vit_ckpt_path,
                                              select_layer_list=model_args.select_layer_list,
                                              select_feature=model_args.select_feature,image_size=model_args.image_size,
                                              patch_size=model_args.patch_size,visual_query_token_nums=model_args.visual_query_token_nums,
                                              audio_query_token_nums=model_args.audio_query_token_nums,BEATs_ckpt_path=model_args.BEATs_ckpt_path,
                                              prompt_embed_dim=model_args.prompt_embed_dim,mask_decoder_transformer_depth=model_args.mask_decoder_transformer_depth,
                                              low_res_mask_size=model_args.low_res_mask_size,
                                              avs_query_num=model_args.avs_query_num,
                                              num_classes=model_args.num_classes,
                                              query_generator_num_layers=model_args.query_generator_num_layers,
                                              dice_loss_weight=training_args.dice_loss_weight,
                                              bce_loss_weight=training_args.bce_loss_weight,)

    ## load ckpt
    if training_args.seg_branch:
        visual_ckpt_dir = 'results/pretrain/llama-visual-qformer/checkpoint-5909'
        ckpt = torch.load(join(visual_ckpt_dir,'non_lora_trainables.bin'),map_location='cpu')
        model.load_state_dict(ckpt,strict=False)
        print('training seg branch, load visual ckpt finished...')
        model.get_model().visual_encoder.requires_grad_(False)
        model.get_model().vl_projector.requires_grad_(False)

    ckpt_dir = 'results/pretrain/llama-seg/checkpoint-2141'
    non_lora_ckpt_path = join(ckpt_dir,'non_lora_trainables.bin')
    ckpt = torch.load(non_lora_ckpt_path,map_location='cpu')
    model.load_state_dict(ckpt,strict=False)
    model.eval()
    model.cuda()
    print('load seg branch ckpt finished...')
    
    image_processor = model.get_model().visual_encoder.image_processor if training_args.visual_branch else None
    dataset, collator = get_dataset_collator(data_args=data_args, tokenizer=tokenizer, 
                                             image_processor=image_processor, mode='test')
    
    model.config.use_cache = True

    ## inference
    dataloader = DataLoader(dataset=dataset,batch_size=1,shuffle=False,collate_fn=collator)
    fp = join(ckpt_dir,'inference_results.jsonl')
    pbar = tqdm(total=len(dataloader),desc='inference')

    for step,sample in enumerate(dataloader):
        batch_metadata = sample.pop('batch_metadata')
        instruction = batch_metadata[0]['instruction']
        label = batch_metadata[0]['output']
        image_path = batch_metadata[0]['image_path']
        mask_path = batch_metadata[0]['mask_path']
        gt_mask = sample['batch_X_modals'][0]['<mask>'] # 1,224,224
        
        sample = prepare_sample(data=sample)
        sample.update(
            {
                # 'do_sample':False,
                # 'top_k':None,
                # 'top_p':None,
                'use_cache':True,
                'max_new_tokens':100
            }
        )
        save_dir = join(ckpt_dir,'mask_img_dir')
        os.makedirs(save_dir,exist_ok=True)
        with torch.no_grad():
            result = model.generate(**sample)
            if result is None:
                print(f'result is None! step:{step} image_path:{image_path} mask_path:{mask_path}')
            else:
                output_ids = result['output_ids']
                output = tokenizer.decode(output_ids[0],skip_special_tokens=False)
                
                pred_masks = result['pred_masks']
                pred_mask = pred_masks[0]
                # mask_scores = result['mask_scores']
                iou = mask_iou(
                    pred=pred_mask.cpu(),
                    target=gt_mask.cpu(),
                )
                print('iou: ',iou)

                pred_mask = (torch.sigmoid(pred_mask)>0.5).int()
                pred_mask = pred_mask.cpu().data.numpy().astype(np.uint8)
                pred_mask = pred_mask*255
                im = Image.fromarray(pred_mask[0]).convert('P')
                filename = mask_path.split('/')[-1][:-4]
                pred_save_path = join(save_dir,f'{filename}_pred.png')
                im.save(pred_save_path, format='PNG')

                gt_mask = (torch.sigmoid(gt_mask)>0.5).int()
                gt_mask = gt_mask.cpu().data.numpy().astype(np.uint8)
                gt_mask = gt_mask*255
                gt = Image.fromarray(gt_mask[0]).convert('P')
                gt_save_path = join(save_dir,f'{filename}_gt.png')
                gt.save(gt_save_path,format='PNG')

                dict_data= {
                    'image_path':image_path,
                    'gt_path':gt_save_path,
                    'pred_path':pred_save_path,
                    'output':output,
                    # 'mask_score':mask_scores[0].item(),
                    'iou':iou.item()
                }

                write2json(fp=fp,dict_data=dict_data)
        pbar.update(1)


if __name__ == "__main__":
    train()



