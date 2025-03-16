import os,sys
sys.path.append(os.getcwd())
from os.path import join,exists
import pathlib
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    print('no npu!')

from torch.utils.data import DataLoader
import transformers
from configs.unified_config import ModelArguments,DataArguments,TrainingArguments,InferenceArguments
from dataset.quick_start_dataset import get_dataset_collator,get_v2_pallete
from utils.util import set_seed,find_all_linear_names,prepare_sample,write2json,load_ckpt
from utils.avss_utils import (
    mask_iou,compute_miou_from_jsonl,calc_color_miou_fscore,
    save_color_mask,save_gt_mask,Eval_Fmeasure,
    metric_s_for_null
)
from utils.deepspeed_utils import *

local_rank = None


def inference_ntp(dataloader,ckpt_dir,model,tokenizer):
    pbar = tqdm(total=len(dataloader),desc=f'inference')
    for step, sample in enumerate(dataloader):
        batch_metadata = sample.pop('batch_metadata')
        bs = len(batch_metadata)
        sample = prepare_sample(data = sample)
        sample.update(
            {
                'use_cache':True,
                'max_new_tokens':500,
            }
        )
        with torch.no_grad():
            output = model.generate(**sample)
            output = tokenizer.batch_decode(output,skip_special_tokens=False)
        for i in range(bs):
            metadata = batch_metadata[i]
            metadata['predict'] = output[i]
            print(metadata)
        pbar.update(1)
    pbar.close()


def inference_ms3(dataloader,ckpt_dir,model,tokenizer):
    save_dir = join(ckpt_dir,f'inference_ms3')
    os.makedirs(save_dir,exist_ok=True)
    pbar = tqdm(total=len(dataloader),desc=f'inference ms3')
    miou = 0.
    f1 = 0.
    count = 0
    fp = join(save_dir,'inference_results.jsonl')
    for step,sample in enumerate(dataloader):
        batch_metadata = sample.pop('batch_metadata')
        sample = prepare_sample(data=sample)
        sample.update(
            {
                'use_cache':True,
                'max_new_tokens':100,
            }
        )
        img_dir = join(save_dir,'mask_img_dir')
        os.makedirs(img_dir,exist_ok=True)
        with torch.no_grad():
            result = model.generate_avs(**sample)
            if result is None:
                pbar.write(f'result is None! step:{step} image_path:{image_path} mask_path:{mask_path}')
            else:
                output_ids = result['output_ids']
                # output = tokenizer.decode(output_ids[0],skip_special_tokens=False)
                output = tokenizer.batch_decode(output_ids,skip_special_tokens=False)
                pred_masks = result.get('pred_masks',None)
                if pred_masks is None:
                    pbar.write(f'step: ',step, '  pred_masks is None.')
                    pbar.update(1)
                    continue
                bs = len(pred_masks)
                for i in range(bs):
                    image_path = batch_metadata[i]['image_path']
                    mask_path = batch_metadata[i]['mask_path']
                    gt_mask = sample['batch_X_modals'][i]['<mask>'] # 1,224,224

                    pred_mask = pred_masks[i] # num_classes,224,224
                    video_name = mask_path.split('/')[-3]
                    os.makedirs(join(img_dir,video_name),exist_ok=True)
                    filename = mask_path.split('/')[-1][:-4]

                    pred_mask_img = (torch.sigmoid(pred_mask)>0.5).int()
                    pred_mask_img = pred_mask_img.cpu().data.numpy().astype(np.uint8)
                    pred_mask_img = pred_mask_img * 255
                    pred_mask_img = Image.fromarray(pred_mask_img[0]).convert('P')
                    pred_save_path = join(img_dir,video_name,f'{filename}_pred.png')
                    pred_mask_img.save(pred_save_path, format='PNG')

                    gt_mask_img = (torch.sigmoid(gt_mask)>0.5).int()
                    gt_mask_img = gt_mask_img.cpu().data.numpy().astype(np.uint8)
                    gt_mask_img = gt_mask_img * 255
                    gt_mask_img = Image.fromarray(gt_mask_img[0]).convert('P')
                    gt_save_path = join(img_dir,video_name,f'{filename}_gt.png')
                    gt_mask_img.save(gt_save_path,format='PNG')

                    image = Image.open(image_path).convert('RGB')
                    image = image.resize((224,224))
                    img_save_path = join(img_dir,video_name,filename+'_image.jpg')
                    image.save(img_save_path)

                    iou = mask_iou(pred=pred_mask.cpu(),target=gt_mask.cpu())
                    fscore = Eval_Fmeasure(pred=pred_mask.cpu(),gt=gt_mask.cpu())
                    miou += iou
                    f1 += fscore
                    count += 1
                    pbar.write(f'iou: {iou} fscore: {fscore}')
                    dict_data= {
                        'image_path':image_path,
                        'gt_path':gt_save_path,
                        'pred_path':pred_save_path,
                        'output':output[i],
                        'iou':iou.item(),
                        'fscore':fscore,
                    }
                    write2json(fp=fp,dict_data=dict_data)
        pbar.update(1)
    pbar.close()
    miou = miou / count
    f1 = f1 / count
    print(f'miou: {miou} f-score: {f1} tot: {count}')


def inference_s4(dataloader,ckpt_dir,model,tokenizer):
    pbar = tqdm(total=len(dataloader),desc=f'inference s4')
    miou = 0.
    f1 = 0.
    count = 0
    for step,sample in enumerate(dataloader):
        batch_metadata = sample.pop('batch_metadata')
        print(batch_metadata)
        sample = prepare_sample(data=sample)
        sample.update(
            {
                'use_cache':True,
                'max_new_tokens':100,
            }
        )
        with torch.no_grad():
            result = model.generate_avs(**sample)
            output_ids = result.get('output_ids',None)
            if output_ids is not None:
                output = tokenizer.batch_decode(output_ids,skip_special_tokens=False)
                print('output: ',output)
            else:
                output = 'none'
            pred_masks = result.get('pred_masks',None)
            if pred_masks is None:
                pbar.write(f'pred_masks is None! step:{step}')
            else:
                bs = len(pred_masks)
                for i in range(bs):
                    image_path = batch_metadata[i]['image_path']
                    mask_path = batch_metadata[i]['mask_path']
                    instruction = batch_metadata[i]['instruction']
                    label = batch_metadata[i]['output']
                    gt_mask = sample['batch_X_modals'][i]['<mask>'] # 1,224,224
                    pred_mask = pred_masks[i] # num_classes,224,224

                    pred_mask_img = (torch.sigmoid(pred_mask)>0.5).int()
                    pred_mask_img = pred_mask_img.cpu().data.numpy().astype(np.uint8)
                    pred_mask_img = pred_mask_img * 255
                    pred_mask_img = Image.fromarray(pred_mask_img[0]).convert('P')
                    pred_save_path = './s4_pred_mask.png'
                    pred_mask_img.save(pred_save_path, format='PNG')

                    gt_mask_img = (torch.sigmoid(gt_mask)>0.5).int()
                    gt_mask_img = gt_mask_img.cpu().data.numpy().astype(np.uint8)
                    gt_mask_img = gt_mask_img * 255
                    gt_mask_img = Image.fromarray(gt_mask_img[0]).convert('P')
                    gt_save_path = './s4_gt_mask.png'
                    gt_mask_img.save(gt_save_path,format='PNG')

                    image = Image.open(image_path).convert('RGB')
                    image = image.resize((224,224))
                    img_save_path = './s4_raw_image.jpg'
                    image.save(img_save_path)

                    iou = mask_iou(pred=pred_mask.cpu(),target=gt_mask.cpu())
                    fscore = Eval_Fmeasure(pred=pred_mask.cpu(),gt=gt_mask.cpu())
                    miou += iou
                    f1 += fscore
                    count += 1
                    pbar.write(f'iou: {iou} fscore: {fscore}')
        pbar.update(1)
    pbar.close()
    miou = miou / count
    f1 = f1 / count
    print(f'miou: {miou} f-score: {f1} tot: {count}')


def inference_ref_avs(dataloader,ckpt_dir,model,tokenizer,test_name='test_s'):
    pbar = tqdm(total=len(dataloader),desc=f'inference ref-avs {test_name}')
    miou = 0.
    f1 = 0.
    count = 0
    for step,sample in enumerate(dataloader):
        batch_metadata = sample.pop('batch_metadata')
        bs = len(batch_metadata)
        sample = prepare_sample(data=sample)
        sample.update(
            {
                'use_cache':True,
                'max_new_tokens':100,
            }
        )
        with torch.no_grad():
            result = model.generate_avs(**sample)
            if result is None:
                pbar.write(f'result is None! step:{step} image_path:{image_path} mask_path:{mask_path}')
            else:
                output_ids = result['output_ids']
                output = tokenizer.batch_decode(output_ids,skip_special_tokens=False)
                pred_masks = result.get('pred_masks',None)
                if pred_masks is None:
                    print(f'step: {step} pred_masks is None')
                    pbar.update(1)
                    continue
            for i in range(bs):
                image_path = batch_metadata[i]['image_path']
                mask_path = batch_metadata[i]['mask_path']
                label = batch_metadata[i]['output']
                gt_mask = sample['batch_X_modals'][i]['<mask>'] # 1,224,224
                pred_mask = pred_masks[i] # num_classes,224,224

                pred_mask_img = (torch.sigmoid(pred_mask)>0.5).int()
                pred_mask_img = pred_mask_img.cpu().data.numpy().astype(np.uint8)
                pred_mask_img = pred_mask_img * 255
                pred_mask_img = Image.fromarray(pred_mask_img[0]).convert('P')
                pred_save_path = './ref-avs_pred.png'
                pred_mask_img.save(pred_save_path, format='PNG')

                gt_mask_img = (torch.sigmoid(gt_mask)>0.5).int()
                gt_mask_img = gt_mask_img.cpu().data.numpy().astype(np.uint8)
                gt_mask_img = gt_mask_img * 255
                gt_mask_img = Image.fromarray(gt_mask_img[0]).convert('P')
                gt_save_path = './ref-avs_gt.png'
                gt_mask_img.save(gt_save_path,format='PNG')

                image = Image.open(image_path).convert('RGB')
                image = image.resize((224,224))
                img_save_path = './ref-avs_raw_image.jpg'
                image.save(img_save_path)

                iou = mask_iou(pred=pred_mask.cpu(),target=gt_mask.cpu())
                fscore = Eval_Fmeasure(pred=pred_mask.cpu(),gt=gt_mask.cpu())
                miou += iou
                f1 += fscore
                count += 1
                pbar.write(f'iou: {iou} fscore: {fscore}')
        pbar.update(1)
    pbar.close()
    miou = miou / count
    f1 = f1 / count
    print(f'miou: {miou} f-score: {f1} tot: {count}')


def inference_ref_avs_null(dataloader,ckpt_dir,model,tokenizer,test_name='test_n'):
    save_dir = join(ckpt_dir,f'inference_ref_avs_{test_name}')
    os.makedirs(save_dir,exist_ok=True)
    pbar = tqdm(total=len(dataloader),desc=f'inference ref-avs')
    ms = 0.
    count = 0
    fp = join(save_dir,f'inference_results.jsonl')
    for step,sample in enumerate(dataloader):
        batch_metadata = sample.pop('batch_metadata')
        image_path = batch_metadata[0]['image_path']
        mask_path = batch_metadata[0]['mask_path']
        gt_mask = sample['batch_X_modals'][0]['<mask>'] # 1,224,224
        sample = prepare_sample(data=sample)
        sample.update(
            {
                'use_cache':True,
                'max_new_tokens':100,
            }
        )
        img_dir = join(save_dir,'mask_img_dir')
        os.makedirs(img_dir,exist_ok=True)
        with torch.no_grad():
            result = model.generate_avs(**sample)
            if result is None:
                pbar.write(f'result is None! step:{step} image_path:{image_path} mask_path:{mask_path}')
            else:
                output_ids = result['output_ids']
                output = tokenizer.decode(output_ids[0],skip_special_tokens=False)
                pred_masks = result.get('pred_masks',None)
                if pred_masks is None:
                    pbar.write(f'step: {step} pred_masks is None.')
                    pbar.update(1)
                    continue

                pred_mask = pred_masks[0] # num_classes,224,224

                video_name = mask_path.split('/')[-3]
                os.makedirs(join(img_dir,video_name),exist_ok=True)
                filename = mask_path.split('/')[-1][:-4]
                fid = mask_path.split('/')[-2]
                pred_mask_img = (torch.sigmoid(pred_mask)>0.5).int()
                pred_mask_img = pred_mask_img.cpu().data.numpy().astype(np.uint8)
                pred_mask_img = pred_mask_img * 255
                pred_mask_img = Image.fromarray(pred_mask_img[0]).convert('P')
                pred_save_path = join(img_dir,video_name,f'{filename}_pred.png')
                pred_mask_img.save(pred_save_path, format='PNG')

                gt_mask_img = (torch.sigmoid(gt_mask)>0.5).int()
                gt_mask_img = gt_mask_img.cpu().data.numpy().astype(np.uint8)
                gt_mask_img = gt_mask_img * 255
                gt_mask_img = Image.fromarray(gt_mask_img[0]).convert('P')
                gt_save_path = join(img_dir,video_name,f'{filename}_gt.png')
                gt_mask_img.save(gt_save_path,format='PNG')

                image = Image.open(image_path).convert('RGB')
                image = image.resize((224,224))
                img_save_path = join(img_dir,video_name,filename+'_image.jpg')
                image.save(img_save_path)

                s = metric_s_for_null(pred_mask.cpu())
                s = s.item()
                ms += s
                count += 1
                pbar.write(f'vname: {video_name} s: {s}')
                dict_data= {
                    'image_path':image_path,
                    'gt_path':gt_save_path,
                    'pred_path':pred_save_path,
                    's': s,
                    'output':output,
                }
                write2json(fp=fp,dict_data=dict_data)
        pbar.update(1)
    pbar.close()
    ms = ms / count
    print(f'ms: {ms} tot: {count}')


def inference_avss(dataloader,ckpt_dir,model,tokenizer):
    v2_pallete = get_v2_pallete(label_to_idx_path='/group/40061/cserdu/data/AVS/label2idx.json',num_cls=71)
    pbar = tqdm(total=len(dataloader),desc=f'inference avss')
    miou = 0.
    # metrics
    N_CLASSES = 71
    miou_pc = torch.zeros((N_CLASSES)) # miou value per class (total sum)
    Fs_pc = torch.zeros((N_CLASSES)) # f-score per class (total sum)
    cls_pc = torch.zeros((N_CLASSES)) # count per class
    for step,sample in enumerate(dataloader):
        batch_metadata = sample.pop('batch_metadata')
        image_path = batch_metadata[0]['image_path']
        mask_path = batch_metadata[0]['mask_path']
        instruction = batch_metadata[0]['instruction']
        label = batch_metadata[0]['output']
        gt_mask = sample['batch_X_modals'][0]['<mask>'] # 1,224,224
        sample = prepare_sample(data=sample)
        dict_data = {
            'instruction':instruction,
            'label':label,
        }
        with torch.no_grad():
            result = model.generate_avs(**sample)
            output_ids = result['output_ids']
            output = tokenizer.decode(output_ids[0],skip_special_tokens=False)
            dict_data['predict'] = output

            pred_masks = result.get('pred_masks',None)
            if pred_masks is None:
                print('pred masks is None')
                pbar.update(1)
                continue

            pred_mask = pred_masks[0] # num_classes,224,224
            _miou_pc, _fscore_pc, _cls_pc, _ = calc_color_miou_fscore(
                pred = pred_mask.cpu().unsqueeze(0),
                target = gt_mask,
                T = 1
            )
            # compute miou, J-measure
            miou_pc += _miou_pc
            cls_pc += _cls_pc
            # compute f-score, F-measure
            Fs_pc += _fscore_pc

            batch_iou = miou_pc / cls_pc
            batch_iou[torch.isnan(batch_iou)] = 0
            batch_iou = torch.sum(batch_iou) / torch.sum(cls_pc != 0)
            batch_fscore = Fs_pc / cls_pc
            batch_fscore[torch.isnan(batch_fscore)] = 0
            batch_fscore = torch.sum(batch_fscore) / torch.sum(cls_pc != 0)
            
            video_name = mask_path.split('/')[-3]
            filename = mask_path.split('/')[-1]  # e.g. 3.png
            save_color_mask(
                pred_masks=pred_mask.unsqueeze(0),
                save_base_path='./avss_result',
                video_name_list=[video_name],
                filename=filename[:-4]+'_pred.png',
                v_pallete=v2_pallete,
                resize=False,
                resized_mask_size=(224,224),
                T=1,
            )
            save_gt_mask(
                gt_masks=gt_mask,
                save_base_path='./avss_result',
                video_name_list=[video_name],
                filename=filename[:-4]+'_gt.png',
                v_pallete=v2_pallete,
                resize=False,
                resized_mask_size=(224,224),
                T=1,
            )
        
        pbar.write(f'iou: {batch_iou} fscore: {batch_fscore}')
        pbar.update(1)
    pbar.close()

    miou_pc = miou_pc / cls_pc
    # print(f"[test miou] {torch.sum(torch.isnan(miou_pc)).item()} classes are not predicted in this batch")
    miou_pc[torch.isnan(miou_pc)] = 0
    miou = torch.mean(miou_pc).item()
    miou_noBg = torch.mean(miou_pc[:-1]).item()
    f_score_pc = Fs_pc / cls_pc
    # print(f"[test fscore] {torch.sum(torch.isnan(f_score_pc)).item()} classes are not predicted in this batch")
    f_score_pc[torch.isnan(f_score_pc)] = 0
    f_score = torch.mean(f_score_pc).item()
    f_score_noBg = torch.mean(f_score_pc[:-1]).item()
    print(f'miou: {miou} miou_noBg: {miou_noBg} f_score: {f_score} f_score_noBg: {f_score_noBg}')


def inference(attn_implementation=None):
    set_seed(42)
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, InferenceArguments))
    model_args, data_args, training_args, infer_args = parser.parse_args_into_dataclasses()
    d_model = 4096
    compute_dtype = torch.float32
    if training_args.fp16:
        compute_dtype = torch.float16
    elif training_args.bf16:
        compute_dtype = torch.bfloat16
    
    pretrain_model_name_or_path = model_args.model_name_or_path
    from models.unified_llama import UnifiedForCausalLM
    from transformers import LlamaConfig
    config = LlamaConfig.from_pretrained(pretrain_model_name_or_path, local_files_only=True)
    config._attn_implementation = attn_implementation
    model = UnifiedForCausalLM.from_pretrained(
        pretrain_model_name_or_path,
        config=config,
        torch_dtype=compute_dtype
    )

    from peft_hyper import LoraConfig,get_peft_model
    lora_trainable="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"
    target_modules = lora_trainable.split(',')
    lora_rank = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_nums = 3
    modules_to_save = None
    peft_config = LoraConfig(
        task_type = "CAUSAL_LM",
        target_modules = target_modules,
        inference_mode = False,
        r = lora_rank, 
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        lora_nums = lora_nums,
        # modules_to_save=modules_to_save
    )
    model = get_peft_model(model, peft_config)

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
    model.get_model().init_multimodal_modules(visual_branch=training_args.visual_branch,
                                              audio_branch=training_args.audio_branch,
                                              segment_branch=training_args.seg_branch,
                                              d_model=d_model,vit_ckpt_path=model_args.vit_ckpt_path,
                                              select_layer_list=model_args.select_layer_list,
                                              select_feature=model_args.select_feature,
                                              image_size=model_args.image_size,
                                              patch_size=model_args.patch_size,
                                              visual_query_token_nums=model_args.visual_query_token_nums,
                                              audio_query_token_nums=model_args.audio_query_token_nums,
                                              BEATs_ckpt_path=model_args.BEATs_ckpt_path,
                                              prompt_embed_dim=model_args.prompt_embed_dim,
                                              mask_decoder_transformer_depth=model_args.mask_decoder_transformer_depth,
                                              low_res_mask_size=model_args.low_res_mask_size,
                                              avs_query_num=model_args.avs_query_num,
                                              num_classes=model_args.num_classes,
                                              query_generator_num_layers=model_args.query_generator_num_layers,
                                              dice_loss_weight=training_args.dice_loss_weight,
                                              bce_loss_weight=training_args.bce_loss_weight,
                                              use_vqgan=False)

    image_scale_nums = model_args.image_scale_nums
    token_nums_per_scale = model_args.token_nums_per_scale
    model.initialize_MM_tokenizer(tokenizer,mask_token_nums = image_scale_nums * token_nums_per_scale, use_vqgan=False)
    MM_tokenizer_vocab_nums = len(tokenizer)
    print('ori_tokenizer_vocab_nums: ',ori_tokenizer_vocab_nums, ' MM_tokenizer_vocab_nums: ',MM_tokenizer_vocab_nums)

    infer_avs = False
    if data_args.ms3_task or data_args.s4_task or data_args.avss_task or data_args.ref_avs_task:
        infer_avs = True
    print('infer_avs task: ',infer_avs)
    ckpt_dir = infer_args.ckpt_dir
    avs_ckpt_dir = infer_args.avs_ckpt_dir
    if not infer_avs:
        ckpt_path = join(ckpt_dir,'finetune_weights.bin')
        ckpt = torch.load(ckpt_path,map_location='cpu')
        model.load_state_dict(ckpt,strict=False)
        print(f'load ckpt from {ckpt_path} finished...')
    else:
        ## hyper lora ckpt
        ckpt_path = join(ckpt_dir,'finetune_weights.bin')
        ckpt = torch.load(ckpt_path,map_location='cpu')
        model.load_state_dict(ckpt,strict=False)
        print(f'load hyper_lora weights from {ckpt_path} finished...')
        ## seg module ckpt
        ckpt_path = join(avs_ckpt_dir,'finetune_weights.bin')
        ckpt = torch.load(ckpt_path,map_location='cpu')
        model.load_state_dict(ckpt,strict=False)
        print(f'load seg_module ckpt from {ckpt_path} finished...')

    device = infer_args.device
    torch.cuda.set_device(device)
    model.npu()
    model.eval()
    image_processor = model.get_model().visual_encoder.image_processor if training_args.visual_branch else None
    dataset, collator = get_dataset_collator(data_args=data_args, tokenizer=tokenizer, 
                                             image_processor=image_processor,mode='test',
                                             test_name=infer_args.test_name)
    
    batch_size = 1
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=False,collate_fn=collator,drop_last=False)
    if data_args.avqa_task or data_args.ave_task or data_args.avvp_task or data_args.arig_task:
        inference_ntp(dataloader=dataloader,ckpt_dir=ckpt_dir,model=model,tokenizer=tokenizer)

    if data_args.s4_task:
        inference_s4(dataloader=dataloader,ckpt_dir=ckpt_dir,model=model,tokenizer=tokenizer)
    
    if data_args.ms3_task:
        inference_ms3(dataloader=dataloader,ckpt_dir=ckpt_dir,model=model,tokenizer=tokenizer)
        
    if data_args.avss_task:
        inference_avss(dataloader=dataloader,ckpt_dir=ckpt_dir,model=model,tokenizer=tokenizer)
    if data_args.ref_avs_task:
        test_name = infer_args.test_name
        if test_name == 'test_n':
            inference_ref_avs_null(dataloader,ckpt_dir,model,tokenizer)
        else:
            inference_ref_avs(dataloader=dataloader,ckpt_dir=ckpt_dir,model=model,tokenizer=tokenizer,test_name=test_name)
            

if __name__ == "__main__":
    inference()


