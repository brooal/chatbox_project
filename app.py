import os
import gradio as gr
from transformers import AutoModelForCausalLM , AutoTokenizer
import torch

# 指定模型下载的目录
MODEL_CACHE_DIR = "./models"
os.makedirs(MODEL_CACHE_DIR , exist_ok = True)

# 选择合适的模型
MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"

# 检查GPU的可用性
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用的设备：{device}")

# 加载模型和分词器（首次运行会下载）
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir = MODEL_CACHE_DIR
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir = MODEL_CACHE_DIR,
    device_map = "auto" if device == "cuda" else None,
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
).eval()


# 模型配置文件(调整以优化性能)
GENERATION_CONFIG = {
    "max_new_tokens" : 512,
    # 是否采用采方法，true使用，False不使用
    # 使用时，会从概率分布中随机采样下一个token
    "do_sample" : True,
    # 值越大，概率分布越平滑；值越小，分布越极端，输出更集中
    "temperature" : 0.7,
    "top_p" : 0.9,
    # 重复惩罚系数，对已经生成的token进行惩罚，避免模型过度重复生成相同的token
    "repetition_penalty" : 1.1,
    # 填充token的id，用于文本序列中填充空白的位置
    # 这里使用的是分词器结束符token的id作为填充的token的id
    "pad_token_id" : tokenizer.eos_token_id
} 

def generate_reponse(message,history):
    """
    处理用户输入并生成回复
    
    Args:
        message (str) : 用户当前输入的信息
        history (str) : 历史对话记录，格式为[(user_msg,bot_msg), ...]
    Returns:
        str:模型生成的回复文本
    
    """
    # 特殊命令处理
    if message.strip() == "/clear":
        return "",[]


    # 格式化为模型需要的聊天格式
    messages = []
    for human , assistant in history :
        messages.append({"role": "user","content":human})
        messages.append({"role": "assistant","content": assistant})
    messages.append({"role":"user","content":message})


    # 使用模型聊天模板格式化输入
    model_inputs = tokenizer.apply_chat_template(
        messages,
        # 添加生成提示符
        add_generation_prompt = True, 
        # 进行分词处理
        tokenizer = True,
        # 返回pytorch张量格式
        return_tensors = "pt"
    ).to(device)

    # 生成回复
    # 关闭梯度计算，节省内存并加速推理
    with torch.no_grad():
        outputs = model.generate(
            model_inputs,
            **GENERATION_CONFIG
        )


    # 解码并返回结果
    # 提取新生成的部分，去掉输入的部分
    response = outputs[0][model_inputs.shape[-1]:]
    # 将token序列解码为文本，跳过特殊token并去除首位空格
    return tokenizer.decode(response,skip_special_tokens=True).strip()

# 创建Gradio界面
chat_interface = gr.ChatInterface(
    fn = generate_reponse,
    title = "MiniChat",
    description = "基于Qwen1.5-0.5B的智能聊天机器人",
    examples = ["你好，你是谁？","用Python写一个快速排序"],
    # retry_btn = None,
    # undo_btn = None,
    # clear_btn = "清空聊天"
)

# 启动应用\
if __name__ =="__main__":
    chat_interface.launch()
