import sys
sys.path.append("/root/autodl-tmp/VQualA/traditional_module/model")

import torch
import torch.nn as nn

from model import Swin2D, Traditionalmodule

model = Traditionalmodule()

for name, param in model.swin.named_parameters():
    print(name)




