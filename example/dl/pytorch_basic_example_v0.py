"""
@brief: This is a basic example of how to use PyTorch to train a model.
@author: 
@date: 2026-01-14
"""

import torch
import torch.nn as nn
import torch.optim as optim

data_2_3 = torch.randn(2, 3)
print("data_1: ", data_2_3)

data_3_2 = data_2_3.permute(1, 0)
print("data_1_1: ", data_3_2)

data_2_3_4 = torch.randn(2, 3, 4)
print("data_2_3_4: ", data_2_3_4)

data_4_2_3 = data_2_3_4.permute(2, 0, 1)
print("data_4_2_3: ", data_4_2_3)

# Hight, Width, Channel
image_2_4_3 = torch.randn(2, 4, 3)
print("image_2_4_3: ", image_2_4_3)

# Channel, Hight, Width
image_3_2_4 = image_2_4_3.permute(2, 0, 1)
print("image_3_2_4: ", image_3_2_4)