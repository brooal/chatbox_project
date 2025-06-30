import os
import gradio as gr
from transformers import AutoModelForCausalLM , AutoTokenizer
import torch

# 选择合适的模型
MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"

# 检查GPU的可用性
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用的设备：{device}")

# 加载模型和分词器（首次运行会下载）
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM(
    MODEL_NAME,
    device_map = "auto" if device == "cuda" else None,
    torch_dtypr = torch.float16 if device == "cuda" else torch.float32
).eavl()


# 模型配置文件(调整以优化性能)
GENERATION_CONFIG = {
    "max_new_token" : 512,
    "do_sample" : True,
    "temperature" : 0.7,
    "top_p" : 0.9,
    "repetition_penalty" : 1.1,
    "pad_token_id" : tokenizer.eos_token_id
} 

def generate_reponse(message,history):
    """处理用户输入并生成回复"""
    

