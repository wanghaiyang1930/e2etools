"""
@brief: This is an example of using timm to get feature map.
@author: wanghaiyang
@date: 2026-01-08
"""

import timm
import torch

from PIL import Image

# Create model with features_only=True (no classifier head)
model = timm.create_model('resnet50',
                          pretrained=False,
                          features_only=True)
# Load checkpoint manually with strict=False to ignore classifier weights
checkpoint = torch.load('/home/workspace/e2e/diffusion/models/resnet50.a1_in1k/pytorch_model.bin', map_location='cpu')
# Remove classifier weights from checkpoint if present
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif isinstance(checkpoint, dict):
    state_dict = checkpoint
else:
    state_dict = checkpoint
# Filter out classifier weights
# Reason: Unexpected key(s) in state_dict: "fc.weight", "fc.bias". 
state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
model.load_state_dict(state_dict, strict=False)

model = model.eval()

data_config = timm.data.resolve_model_data_config(model)
print(data_config)

transforms = timm.data.create_transform(**data_config, is_training=False)

img = Image.open('./beignets-task-guide.png')
print("image size: ", img.size)

input = transforms(img).unsqueeze(0)
print("input shape: ", input.shape)

output = model(input)

for layer in output:
    print("output layer shape: ", layer.shape)

