"""
@brief: This is a basic example of how to use PyTorch to train a model.
@author: 
@date: 2026-01-14
"""


import torch
import torchvision

from PIL import Image

image = Image.open("beignets-task-guide.png")

image_transform = torchvision.transforms.ToTensor()

image_data = image_transform(image)

image_datas = image_data.unsqueeze(0)

print("image size: ", image.size)
print("image data shape: ", image_data.shape)
print("image data patch shape: ", image_datas.shape)

