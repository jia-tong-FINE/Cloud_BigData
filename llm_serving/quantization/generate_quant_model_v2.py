# generate quant model by autogptq

from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

pretrained_model_dir = '/benchmark/Llama-2-7b-hf'
quantized_model_dir = '/benchmark/Llama-2-7b-4bit'


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)

print('Done with tokenizer...')

examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

quantize_config = BaseQuantizeConfig(
    bits=4,  # 将模型量化为 4-bit 数值类型
    group_size=128,  # 一般推荐将此参数的值设置为 128
    desc_act=False,  # 设为 False 可以显著提升推理速度，但是 ppl 可能会轻微地变差
)

# 加载未量化的模型，默认情况下，模型总是会被加载到 CPU 内存中
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

print('Done with model...')

# 量化模型, 样本的数据类型应该为 List[Dict]，其中字典的键有且仅有 input_ids 和 attention_mask
model.quantize(examples)

print('Done with quantize...')

# 保存量化好的模型
model.save_quantized(quantized_model_dir)

print('Done with save...')

# 使用 safetensors 保存量化好的模型
model.save_quantized(quantized_model_dir, use_safetensors=True)

# 将量化好的模型直接上传至 Hugging Face Hub 
# 当使用 use_auth_token=True 时, 确保你已经首先使用 huggingface-cli login 进行了登录
# 或者可以使用 use_auth_token="hf_xxxxxxx" 来显式地添加账户认证 token
# （取消下面三行代码的注释来使用该功能）
# repo_id = f"YourUserName/{quantized_model_dir}"
# commit_message = f"AutoGPTQ model for {pretrained_model_dir}: {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"
# model.push_to_hub(repo_id, commit_message=commit_message, use_auth_token=True)

# 或者你也可以同时将量化好的模型保存到本地并上传至 Hugging Face Hub
# （取消下面三行代码的注释来使用该功能）
# repo_id = f"YourUserName/{quantized_model_dir}"
# commit_message = f"AutoGPTQ model for {pretrained_model_dir}: {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"
# model.push_to_hub(repo_id, save_dir=quantized_model_dir, use_safetensors=True, commit_message=commit_message, use_auth_token=True)

# 加载量化好的模型到能被识别到的第一块显卡中
# model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0")

# 从 Hugging Face Hub 下载量化好的模型并加载到能被识别到的第一块显卡中
# model = AutoGPTQForCausalLM.from_quantized(repo_id, device="cuda:0", use_safetensors=True, use_triton=False)

# 使用 model.generate 执行推理
# print(tokenizer.decode(model.generate(**tokenizer("auto_gptq is", return_tensors="pt").to(model.device))[0]))

# 或者使用 TextGenerationPipeline
# pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
# print(pipeline("auto-gptq is")[0]["generated_text"])