

# DDIMScheduler 使用指南

## 简介

`DDIMScheduler` 是 `diffusers` 库中用于扩散模型采样的调度器。DDIM（Denoising Diffusion Implicit Models）相比传统的 DDPM，可以用更少的采样步数（如 10-50 步）生成高质量样本。

## 安装

```bash
pip install diffusers torch numpy
```

## 核心概念

### 1. 时间步（Timesteps）
- **训练时间步数** (`num_train_timesteps`): 训练时使用的总步数，通常为 1000
- **推理时间步数** (`num_inference_steps`): 生成时使用的步数，可以远少于训练步数（DDIM的优势）
- **时间步序列**: 从大到小，如 `[999, 950, 900, ..., 50, 0]`

### 2. 噪声调度（Beta Schedule）
控制噪声如何随时间步增加：
- `linear`: 线性增加
- `scaled_linear`: 缩放线性
- `squaredcos_cap_v2`: 余弦平方（常用）

### 3. Eta 参数
控制采样的随机性：
- `eta=0`: 确定性采样（纯DDIM，可复现）
- `eta=1`: 随机采样（DDPM风格）
- `0<eta<1`: 介于两者之间

## 基本用法

### 步骤1: 初始化调度器

```python
from diffusers import DDIMScheduler

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear",
    clip_sample=True,  # 是否将样本裁剪到 [-1, 1]
)
```

### 步骤2: 设置推理步数

```python
scheduler.set_timesteps(num_inference_steps=50)
```

### 步骤3: 初始化噪声

```python
from diffusers.utils.torch_utils import randn_tensor

shape = (batch_size, channels, height, width)
latents = randn_tensor(shape)
```

### 步骤4: 逐步去噪生成

```python
for t in scheduler.timesteps:
    # 模型预测噪声
    noise_pred = model(latents, t)
    
    # 调度器执行一步去噪
    scheduler_output = scheduler.step(
        model_output=noise_pred,
        timestep=t,
        sample=latents,
        return_dict=True
    )
    
    latents = scheduler_output.prev_sample
```

## 完整示例

### 示例1: 基本使用

```python
import torch
from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor

# 初始化
scheduler = DDIMScheduler(num_train_timesteps=1000)
scheduler.set_timesteps(num_inference_steps=20)

# 从噪声开始
latents = randn_tensor((1, 3, 64, 64))

# 去噪循环
for t in scheduler.timesteps:
    noise_pred = model(latents, t)  # 你的模型
    latents = scheduler.step(noise_pred, t, latents, return_dict=True).prev_sample
```

### 示例2: 前向扩散（添加噪声）

```python
# 创建原始数据
original_sample = torch.randn(1, 3, 64, 64)

# 选择时间步
timestep = 500

# 添加噪声
noise = randn_tensor(original_sample.shape)
noisy_sample = scheduler.add_noise(original_sample, noise, timestep)
```

### 示例3: 自定义参数

```python
# 快速采样（步数少）
scheduler.set_timesteps(num_inference_steps=10)

# 高质量采样（步数多）
scheduler.set_timesteps(num_inference_steps=50)

# 控制随机性
scheduler.step(noise_pred, t, latents, eta=0.0, return_dict=True)  # 确定性
scheduler.step(noise_pred, t, latents, eta=1.0, return_dict=True)  # 随机
```

## 关键方法

### `set_timesteps(num_inference_steps)`
设置推理时的采样步数。DDIM的优势是可以使用远少于训练步数的步数。

### `step(model_output, timestep, sample, eta=0.0, return_dict=True)`
执行一步去噪：
- `model_output`: 模型预测的噪声
- `timestep`: 当前时间步
- `sample`: 当前样本
- `eta`: 随机性参数（0-1）
- 返回: `DDIMSchedulerOutput` 或元组，包含 `prev_sample`

### `add_noise(original_samples, noise, timesteps)`
前向扩散：向原始样本添加噪声。

## 常见问题

### Q: 推理步数应该设置多少？
A: 通常 10-50 步即可。更多步数质量更好但速度更慢，可以根据需求平衡。

### Q: eta 参数如何选择？
A: 
- `eta=0`: 确定性，适合需要可复现结果的场景
- `eta=0.5-1.0`: 增加多样性，适合生成多样化样本

### Q: 如何与 UNet 等模型配合使用？
A: 
```python
for t in scheduler.timesteps:
    # UNet 预测噪声
    with torch.no_grad():
        noise_pred = unet(latents, t).sample
    
    # 调度器去噪
    latents = scheduler.step(noise_pred, t, latents).prev_sample
```

## 参考文件

- `ddim_scheduler_example.py`: 完整示例，包含6个详细示例
- `ddim_scheduler_simple.py`: 最简示例，快速上手

## 更多资源

- [Diffusers 官方文档](https://huggingface.co/docs/diffusers)
- [DDIM 论文](https://arxiv.org/abs/2010.02502)
