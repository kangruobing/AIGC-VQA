import torch
import ImageReward as RM


model = RM.load("ImageReward-v1.0")

for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

torch.save(model.state_dict(), "/root/autodl-tmp/VQualA/alignment_module/imagereward.pth")
