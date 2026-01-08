"""
@brief: 
@author: wanghaiyang
@date: 2026-01-08
"""

import timm
import torch
import requests

from PIL import Image

model = timm.create_model('resnet50', 
                          pretrained=False,
                          checkpoint_path='/home/workspace/e2e/diffusion/models/resnet50.a1_in1k/pytorch_model.bin')
model = model.eval()

data_config = timm.data.resolve_model_data_config(model)
print(data_config)

transforms = timm.data.create_transform(**data_config, is_training=False)


img = Image.open('./beignets-task-guide.png')

print("img shape: ", img.size)

input = transforms(img).unsqueeze(0)
output = model(transforms(img).unsqueeze(0))
print("input shpae: ", input.shape)
print("output shape: ", output.shape)

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1), k=5)
print("top5_probabilities: ", top5_probabilities)
print("top5_class_indices: ", top5_class_indices)

# 获取 ImageNet 类别标签
IMAGENET_1K_URL = 'https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt'
response = requests.get(IMAGENET_1K_URL)
imagenet_classes = response.text.strip().split('\n')

# 打印 Top-5 预测结果及其类别名称
print("\nTop-5 预测结果:")
for i in range(5):
    idx = top5_class_indices[0][i].item()
    prob = top5_probabilities[0][i].item()
    class_name = imagenet_classes[idx]
    print(f"{i+1}. [{idx}] {class_name}: {prob:.4f}")



