"""
@author wanghaiyang
@date 2026-01-25

DDIMScheduler 核心用法 - 最简示例
==================================

这是 DDIMScheduler 最核心的使用方法，适合快速上手。
"""

import torch
import numpy as np

from PIL import Image

from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor


def save_latents_as_image(latents, filepath):
    """
    将 latents tensor 保存为图片
    
    Args:
        latents: torch.Tensor, 形状为 (batch_size, channels, height, width)
        filepath: str, 保存路径
    """
    # 转换为 numpy 数组
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()
    
    # 取第一个样本（batch dimension）
    img = latents[0]  # (channels, height, width)
    
    # 将 CHW 转换为 HWC
    img = np.transpose(img, (1, 2, 0))  # (height, width, channels)
    
    # 归一化到 [0, 255]
    # 先归一化到 [0, 1]，然后缩放到 [0, 255]
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    img = (img * 255).astype(np.uint8)
    
    # 保存为图片
    Image.fromarray(img).save(filepath)
    print(f"图片已保存到: {filepath}")


def simple_example():
    """最简单的使用示例"""
    
    # 1. 初始化调度器
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,  # 训练时的总步数
        beta_schedule="linear",     # 噪声调度方式
    )
    
    # 2. 设置推理时的采样步数（DDIM的优势：可以用很少的步数）
    scheduler.set_timesteps(num_inference_steps=20)
    
    # 3. 从纯噪声开始
    shape = (1, 3, 64, 64)  # (batch_size, channels, height, width)
    latents = randn_tensor(shape)
    
    # 保存去噪前的初始噪声
    save_latents_as_image(latents, "ddim_noise_before.png")
    
    # 4. 逐步去噪生成
    for t in scheduler.timesteps:
        # 这里应该调用你的扩散模型来预测噪声
        # 为了演示，我们用一个简单的函数模拟
        noise_pred = predict_noise(latents, t)  # 你的模型: model(latents, t)
        
        # 调度器执行一步去噪
        latents = scheduler.step(
            model_output=noise_pred,  # 模型预测的噪声
            timestep=t,               # 当前时间步
            sample=latents,           # 当前样本
            return_dict=True
        ).prev_sample
    
    # 保存去噪后的最终结果
    save_latents_as_image(latents, "ddim_result_after.png")
    
    return latents


def predict_noise(sample, timestep):
    """
    模拟的噪声预测函数
    实际应用中，这里应该是你的扩散模型（如UNet）
    """
    # 简化示例：返回样本的一部分作为"预测的噪声"
    return sample * 0.1


def example_with_eta():
    """使用 eta 参数控制随机性"""
    
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(num_inference_steps=20)
    
    # 在扩散模型中，latents 通常指潜在空间中的表示，而不是原始像素空间，在纯像素空间的扩散模型中（如你的示例），
    # 更准确的术语是 sample（样本）；
    # 当前示例没有 VAE，直接在像素空间操作，使用 latents 是沿用 diffusers 库的习惯命名，但不够准确，
    # 更准确的命名应该是 sample 或 noisy_image；
    latents = randn_tensor((1, 3, 64, 64))
    
    for t in scheduler.timesteps:
        noise_pred = predict_noise(latents, t)
        
        # eta=0: 确定性采样（纯DDIM，可复现）
        # eta=1: 随机采样（DDPM风格）
        # 0<eta<1: 介于两者之间
        latents = scheduler.step(
            model_output=noise_pred,
            timestep=t,
            sample=latents,
            eta=0.0,  # 设置为0表示确定性采样
            return_dict=True
        ).prev_sample
    
    return latents


if __name__ == "__main__":
    print("运行简单示例...")
    result = simple_example()
    print(f"生成完成！结果形状: {result.shape}")
    
    print("\n运行带eta参数的示例...")
    result2 = example_with_eta()
    print(f"生成完成！结果形状: {result2.shape}")
