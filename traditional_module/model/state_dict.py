import torch

state_dict = torch.load("/root/autodl-tmp/VQualA/traditional_module/model/Swin_b_384_in22k_SlowFast_Fast_LSVQ.pth")

for key, value in state_dict.items():
    print(key)