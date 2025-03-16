import os,sys
sys.path.append(os.getcwd())
from os.path import join
import pathlib
from tqdm import tqdm
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    print('no npu!')

from torch.utils.data import DataLoader
import transformers

from configs.unified_config import ModelArguments,DataArguments,TrainingArguments

from dataset.pretrain_dataset import get_dataset_collator
from utils.util import set_seed,rank0_print,find_all_linear_names,prepare_sample,write2txt,write2json
from utils.deepspeed_utils import *
from models.multimodal_encoder import VisualEncoder,AudioEncoder

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
    print('compute dtype: ',compute_dtype)

    if training_args.train_audio_branch:
        # import deepspeed
        # from transformers.utils import ContextManagers
        # from transformers.integrations import deepspeed_config
        # init_contexts = [deepspeed.zero.Init(config_dict_or_path=deepspeed_config())]
        # with ContextManagers(init_contexts):
        #     audio_encoder = AudioEncoder(d_model=d_model)

        audio_encoder = AudioEncoder(d_model=d_model)

    if training_args.train_visual_branch:
        # visual_encoder = VisualEncoder(d_model=d_model)
        import deepspeed
        from transformers.utils import ContextManagers
        from transformers.integrations import deepspeed_config
        init_contexts = [deepspeed.zero.Init(config_dict_or_path=deepspeed_config())]
        with ContextManagers(init_contexts):
            visual_encoder = VisualEncoder.from_pretrained(d_model=d_model)

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
        rank0_print(local_rank, "Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)


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
    
    model.get_model().pad_token_id = tokenizer.pad_token_id

    if training_args.train_visual_branch:
        model.get_model().visual_encoder = visual_encoder
    if training_args.train_audio_branch:
        model.get_model().audio_encoder = audio_encoder
    
    model.initialize_MM_tokenizer(tokenizer,add_image_tokens=True, add_video_tokens=True, 
                                  add_audio_tokens=True, add_mask_tokens=False)
    
    
    ckpt_path = 'results/pretrain/qwen-audio-2/checkpoint-1017/pretrain_weights.bin'
    ckpt = torch.load(ckpt_path,map_location='cpu')
    model.load_state_dict(ckpt,strict=False)
    model.cuda()
    model.eval()

    # image_processor=visual_encoder.image_processor
    dataset, collator = get_dataset_collator(data_args=data_args, tokenizer=tokenizer, 
                                             image_processor=None, mode='test')
    
    model.config.use_cache = True
    dataloader = DataLoader(dataset=dataset,batch_size=1,shuffle=False,collate_fn=collator)
    
    fp = 'results/pretrain/qwen-audio-2/checkpoint-1017/inference_results.jsonl'
    pbar = tqdm(total=len(dataloader),desc='inference')

    for step,sample in enumerate(dataloader):
        batch_metadata = sample.pop('batch_metadata',None)
        label = batch_metadata[0]['output']
        audio_path = batch_metadata[0]['audio_path']
        sample = prepare_sample(data=sample,dtype=compute_dtype)
        sample.update(
            {
                # 'do_sample':False,
                # 'top_k':None,
                # 'top_p':None,
                'use_cache':True,
                'max_new_tokens':200
            }
        )
        with torch.no_grad():
            output = model.generate(**sample)
        output = tokenizer.decode(output[0],skip_special_tokens=True)
        # info = f'label: {label} output: {output}'
        # write2txt(fp,info)
        data = {
            'label':label,
            'output':output,
            'audio_path':audio_path,
        }
        write2json(fp=fp,dict_data=data)

        pbar.update(1)

    
if __name__ == "__main__":
    train()



