# generate quant model by bitsandbytes in transformers


from transformers import LlamaConfig,LlamaForCausalLM,LlamaTokenizer,BitsAndBytesConfig, AutoModelForCausalLM, GPTQConfig, AutoTokenizer
# from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model
import torch
import tensorflow as tf 

model_path = '/benchmark/Llama-2-7b-hf'

# tokenizer = LlamaTokenizer.from_pretrained(model_path)
# memory_bound = {0: '0GiB', 1: '6GiB', 2: '1GiB', 3: '9GiB'}

# config = LlamaConfig.from_pretrained(model_path)
# with init_empty_weights():
#    model = LlamaForCausalLM._from_config(config, torch_dtype=torch.float16)

# device_map = infer_auto_device_map(model, max_memory=memory_bound, no_split_module_classes=LlamaForCausalLM._no_split_modules)
# load_checkpoint_in_model(model, model_path, device_map=device_map)
# model = dispatch_model(model,device_map=device_map)

quantization_config=BitsAndBytesConfig(
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    device_map='auto')


import os
output = '/benchmark/Llama-2-7b-4bit'

if not os.path.exists(output):
    os.mkdir(output)

model.save_pretrained(output)
print('Done')
