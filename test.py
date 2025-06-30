import torch
print(torch.__version__)
import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}") if torch.cuda.is_available() else print("无CUDA支持")
print(f"设备数量: {torch.cuda.device_count()}")
print(f"当前设备: {torch.cuda.current_device() if torch.cuda.is_available() else '无'}")
print(f"设备名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无'}")