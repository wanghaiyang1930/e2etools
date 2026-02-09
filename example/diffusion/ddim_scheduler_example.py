"""
@author wanghaiyang
@date 2026-01-25

DDIMScheduler 使用示例
=====================

DDIMScheduler 是扩散模型（Diffusion Model）中的调度器，用于控制去噪过程。
相比 DDPM，DDIM 可以使用更少的采样步数（如 10-50 步）生成高质量样本。

安装依赖:
    pip install diffusers torch numpy

本示例展示了：
1. DDIMScheduler 的基本初始化
2. 如何设置采样步数
3. 如何进行前向扩散（添加噪声）
4. 如何进行反向去噪（采样生成）
5. 如何自定义采样参数
6. 理解 scheduler.step() 的输出

核心概念:
- num_train_timesteps: 训练时的总时间步数（通常1000）
- num_inference_steps: 推理时的采样步数（可以远少于训练步数，这是DDIM的优势）
- timestep: 当前的时间步，从大到小（如 999, 998, ..., 0）
- model_output: 模型预测的噪声
- eta: 控制随机性的参数，eta=0 是确定性DDIM，eta=1 是随机DDPM
"""


import torch
import numpy as np

from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor



def example_1_basic_usage():
    """示例1: DDIMScheduler 基本用法"""
    print("=" * 60)
    print("示例1: DDIMScheduler 基本用法")
    print("=" * 60)
    
    # 1. 初始化 DDIMScheduler
    # num_train_timesteps: 训练时的总时间步数（通常为1000）
    # beta_schedule: 噪声调度方式，可选 'linear', 'scaled_linear', 'squaredcos_cap_v2' 等
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=True,  # 是否将样本裁剪到 [-1, 1]
    )
    
    # 2. 设置推理时的采样步数（可以少于训练步数，这是DDIM的优势）
    scheduler.set_timesteps(num_inference_steps=50)
    
    print(f"训练时间步数: {scheduler.config.num_train_timesteps}")
    print(f"推理时间步数: {len(scheduler.timesteps)}")
    print(f"推理时间步序列: {scheduler.timesteps[:10]}...")  # 显示前10个时间步
    
    # 3. 创建一个简单的噪声样本（模拟模型输出）
    batch_size = 2
    sample_shape = (batch_size, 3, 64, 64)  # 假设是图像数据
    sample = randn_tensor(sample_shape, generator=None)
    
    print(f"\n样本形状: {sample.shape}")
    print(f"样本范围: [{sample.min():.3f}, {sample.max():.3f}]")
    
    return scheduler, sample


def example_2_forward_diffusion():
    """示例2: 前向扩散过程（添加噪声）"""
    print("\n" + "=" * 60)
    print("示例2: 前向扩散过程（添加噪声）")
    print("=" * 60)
    
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
    )
    scheduler.set_timesteps(num_inference_steps=50)
    
    # 创建原始数据（模拟）
    batch_size = 1
    original_sample = torch.randn(batch_size, 3, 32, 32)
    
    # 选择一个时间步进行前向扩散
    # timestep 的含义：
    #   - 在扩散模型中，前向过程是从原始数据逐步添加噪声的过程
    #   - timestep 表示当前处于扩散过程的哪个时间步
    #   - timestep 的范围是 0 到 num_train_timesteps-1（这里是 0-999）
    #   - timestep=0: 几乎没有噪声，接近原始数据
    #   - timestep=999: 几乎完全是噪声
    #   - timestep 越大，添加的噪声越多
    # 为什么选择 500？
    #   - 因为 num_train_timesteps=1000，500 是中间值
    #   - 选择中间值可以展示一个中等程度的加噪效果
    #   - 这样可以看到原始数据和完全噪声之间的中间状态
    timestep = 500  # 选择中间的时间步（1000 的一半）
    
    # 添加噪声
    noise = randn_tensor(original_sample.shape, generator=None)
    # 将 timestep 转换为张量（新版本的 diffusers 要求）
    timesteps = torch.tensor([timestep], device=original_sample.device)
    noisy_sample = scheduler.add_noise(original_sample, noise, timesteps)
    
    print(f"原始样本形状: {original_sample.shape}")
    print(f"噪声形状: {noise.shape}")
    print(f"加噪后样本形状: {noisy_sample.shape}")
    print(f"原始样本均值: {original_sample.mean():.4f}")
    print(f"加噪后样本均值: {noisy_sample.mean():.4f}")
    
    return scheduler, original_sample, noisy_sample, timestep


def example_3_reverse_diffusion():
    """示例3: 反向去噪过程（采样生成）"""
    print("\n" + "=" * 60)
    print("示例3: 反向去噪过程（采样生成）")
    print("=" * 60)
    
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
    )
    
    # 设置推理步数（DDIM的优势：可以用更少的步数）
    num_inference_steps = 20
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    
    print(f"推理步数: {num_inference_steps}")
    print(f"时间步序列: {scheduler.timesteps}")
    
    # 1. 从纯噪声开始
    batch_size = 1
    shape = (batch_size, 3, 32, 32)
    latents = randn_tensor(shape, generator=None)
    
    print(f"\n初始噪声形状: {latents.shape}")
    print(f"初始噪声统计: mean={latents.mean():.4f}, std={latents.std():.4f}")
    
    # 2. 逐步去噪
    for i, t in enumerate(scheduler.timesteps):
        # 模拟模型预测的噪声（实际应用中，这里应该调用你的扩散模型）
        # 这里我们用一个简单的函数来模拟
        noise_pred = latents * 0.1  # 简化示例，实际应该是模型输出
        
        # 使用 scheduler.step() 进行一步去噪
        scheduler_output = scheduler.step(
            model_output=noise_pred,
            timestep=t,
            sample=latents,
            return_dict=True
        )
        
        latents = scheduler_output.prev_sample
        
        if i % 5 == 0 or i == len(scheduler.timesteps) - 1:
            print(f"步骤 {i+1}/{len(scheduler.timesteps)}, 时间步 {t.item()}, "
                  f"样本均值: {latents.mean():.4f}, 标准差: {latents.std():.4f}")
    
    print(f"\n最终样本形状: {latents.shape}")
    return scheduler, latents


def example_4_with_model():
    """示例4: 结合简单模型使用"""
    print("\n" + "=" * 60)
    print("示例4: 结合简单模型使用")
    print("=" * 60)
    
    # 初始化调度器
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
    )
    scheduler.set_timesteps(num_inference_steps=10)
    
    # 创建一个简单的"模型"（实际应用中这里应该是你的UNet等）
    class SimpleNoisePredictor(torch.nn.Module):
        """简单的噪声预测器（仅用于演示）"""
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
        
        def forward(self, sample, timestep):
            # 实际模型应该接受 (batch, channels, height, width) 和 timestep
            # 这里简化处理
            return sample * 0.5  # 返回预测的噪声
    
    model = SimpleNoisePredictor()
    
    # 生成过程
    batch_size = 1
    shape = (batch_size, 3, 32, 32)
    latents = randn_tensor(shape, generator=None)
    
    print("开始生成过程...")
    for i, t in enumerate(scheduler.timesteps):
        # 模型预测噪声
        with torch.no_grad():
            noise_pred = model(latents, t)
        
        # 调度器执行一步去噪
        scheduler_output = scheduler.step(
            model_output=noise_pred,
            timestep=t,
            sample=latents,
            return_dict=True
        )
        
        latents = scheduler_output.prev_sample
        
        if i < 3 or i == len(scheduler.timesteps) - 1:
            print(f"步骤 {i+1}: 时间步={t.item()}, "
                  f"预测噪声范围=[{noise_pred.min():.3f}, {noise_pred.max():.3f}]")
    
    print("生成完成！")
    return scheduler, latents


def example_5_custom_parameters():
    """示例5: 自定义参数"""
    print("\n" + "=" * 60)
    print("示例5: 自定义参数")
    print("=" * 60)
    
    # 不同的配置选项
    configs = {
        "快速采样": {
            "num_inference_steps": 10,
            "eta": 0.0,  # eta=0 是纯DDIM，eta=1 是DDPM
        },
        "高质量采样": {
            "num_inference_steps": 50,
            "eta": 0.0,
        },
        "随机性采样": {
            "num_inference_steps": 20,
            "eta": 0.5,  # 增加随机性
        }
    }
    
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
    )
    
    for name, config in configs.items():
        print(f"\n{name}:")
        scheduler.set_timesteps(num_inference_steps=config["num_inference_steps"])
        
        latents = randn_tensor((1, 3, 32, 32), generator=None)
        
        for i, t in enumerate(scheduler.timesteps):
            noise_pred = latents * 0.1
            
            scheduler_output = scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=latents,
                eta=config["eta"],  # 控制随机性
                return_dict=True
            )
            
            latents = scheduler_output.prev_sample
        
        print(f"  步数: {config['num_inference_steps']}, "
              f"eta: {config['eta']}, "
              f"最终均值: {latents.mean():.4f}")


def example_6_scheduler_output():
    """示例6: 理解 scheduler.step() 的输出"""
    print("\n" + "=" * 60)
    print("示例6: 理解 scheduler.step() 的输出")
    print("=" * 60)
    
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
    )
    scheduler.set_timesteps(num_inference_steps=5)
    
    latents = randn_tensor((1, 3, 32, 32), generator=None)
    noise_pred = latents * 0.1
    t = scheduler.timesteps[0]
    
    # 使用 return_dict=True 返回字典格式
    output_dict = scheduler.step(
        model_output=noise_pred,
        timestep=t,
        sample=latents,
        return_dict=True
    )
    
    print("scheduler.step() 返回的对象类型:", type(output_dict))
    print("输出属性:")
    print(f"  - prev_sample: {output_dict.prev_sample.shape}")
    print(f"  - pred_original_sample: {output_dict.pred_original_sample.shape if hasattr(output_dict, 'pred_original_sample') else 'N/A'}")
    
    # 使用 return_dict=False 返回元组
    prev_sample_tuple = scheduler.step(
        model_output=noise_pred,
        timestep=t,
        sample=latents,
        return_dict=False
    )
    
    print(f"\nreturn_dict=False 时返回元组: {type(prev_sample_tuple)}")
    print(f"元组长度: {len(prev_sample_tuple)}")
    print(f"第一个元素形状: {prev_sample_tuple[0].shape}")


def main():
    """运行所有示例"""
    
    print("\n" + "=" * 60)
    print("DDIMScheduler 完整使用示例")
    print("=" * 60)
    
    # 运行各个示例
    example_1_basic_usage()
    example_2_forward_diffusion()
    example_3_reverse_diffusion()
    example_4_with_model()
    example_5_custom_parameters()
    example_6_scheduler_output()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
    
    print("\n关键要点总结:")
    print("1. DDIMScheduler 可以用更少的推理步数（如10-50步）生成样本")
    print("2. scheduler.set_timesteps() 设置推理时的采样步数")
    print("3. scheduler.step() 执行一步去噪，需要模型预测的噪声")
    print("4. eta 参数控制随机性：eta=0 是确定性DDIM，eta=1 是随机DDPM")
    print("5. 前向过程用 add_noise()，反向过程用 step()")
    print("\n典型使用流程:")
    print("  1. 初始化: scheduler = DDIMScheduler(...)")
    print("  2. 设置步数: scheduler.set_timesteps(num_inference_steps=50)")
    print("  3. 初始化噪声: latents = randn_tensor(shape)")
    print("  4. 循环去噪: for t in scheduler.timesteps:")
    print("     - noise_pred = model(latents, t)")
    print("     - latents = scheduler.step(noise_pred, t, latents).prev_sample")


if __name__ == "__main__":
    main()
