"""
@brief: This is an example of using timm to train a model.
@author: wanghaiyang
@date: 2026-01-08
"""

import timm

model = timm.create_model('resnet50', 
                          pretrained=False, 
                          checkpoint_path='/home/workspace/e2e/diffusion/models/resnet50.a1_in1k/pytorch_model.bin')
config = model.default_cfg
print(config)
