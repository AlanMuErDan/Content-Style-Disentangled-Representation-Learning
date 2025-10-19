#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将潜在编码解码为PNG图像进行可视化
用于检查数据内容是否为字体图像
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import yaml
from typing import Optional, Tuple
from torchvision import transforms

# 添加项目根目录到路径
sys.path.append('/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/Content-Style-Disentangled-Representation-Learning')
from models import build_decoder

# VAE解码器相关函数
def _load_config(path: str) -> dict:
    """加载配置文件"""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["vae"] if "vae" in cfg else cfg

def load_vae_decoder(
    config_path: str = "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/Content-Style-Disentangled-Representation-Learning/configs/config.yaml",
    ckpt_path: str = "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0/vae_best_ckpt.pth",
    device: str = "cuda"
) -> Tuple[torch.nn.Module, dict, torch.device]:
    """
    加载VAE解码器
    
    Returns:
        decoder  – torch.nn.Module (eval mode)
        cfg      – dict (配置)
        device   – torch.device
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    cfg = _load_config(config_path)

    decoder = build_decoder(
        name=cfg["decoder"],
        img_size=cfg["img_size"],
        latent_channels=cfg.get("latent_channels", 4),
    ).to(device).eval()

    # 加载权重
    ckpt = torch.load(ckpt_path, map_location=device)
    if "decoder" in ckpt:
        decoder.load_state_dict(ckpt["decoder"])
    else:
        # 如果checkpoint直接是decoder状态
        decoder.load_state_dict(ckpt)

    print(f"[INFO] Loaded VAE decoder on {device}")
    return decoder, cfg, device

def load_latents_from_local(local_path: str, map_location: str = "cpu"):
    """从本地文件加载潜在编码 - 多种尝试方式"""
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")
    
    print(f"[INFO] Loading from local file: {local_path}")
    
    # 检查文件基本信息
    file_size = os.path.getsize(local_path)
    print(f"[INFO] File size: {file_size / (1024*1024):.2f} MB")
    
    # 尝试多种加载方式
    methods = [
        ("Standard torch.load", lambda: torch.load(local_path, map_location=map_location)),
        ("Weights only", lambda: torch.load(local_path, map_location=map_location, weights_only=True)),
        ("With pickle protocol", lambda: torch.load(local_path, map_location=map_location, pickle_protocol=2)),
    ]
    
    for method_name, load_func in methods:
        try:
            print(f"[INFO] Trying {method_name}...")
            obj = load_func()
            print(f"[INFO] Successfully loaded with {method_name}")
            return obj
        except Exception as e:
            print(f"[WARN] {method_name} failed: {e}")
            continue
    
    # 如果所有方法都失败，尝试读取原始数据
    try:
        print(f"[INFO] Trying to read raw data...")
        with open(local_path, 'rb') as f:
            data = f.read(1000)  # 读取前1000字节
            print(f"[INFO] File header (hex): {data[:50].hex()}")
            print(f"[INFO] File header (ascii): {data[:50]}")
    except Exception as e:
        print(f"[WARN] Cannot read raw data: {e}")
    
    raise RuntimeError(f"All loading methods failed for {local_path}")

@torch.no_grad()
def decode_to_image(decoder: nn.Module, z: torch.Tensor, cfg: dict, device: torch.device) -> torch.Tensor:
    """
    使用VAE解码器将潜在编码解码为图像
    
    Args:
        decoder: VAE解码器
        z: 潜在编码 [C,H,W] 或其他格式
        cfg: 配置字典
        device: 计算设备
        
    Returns:
        img: 图像张量 [C,H,W] 范围[0,1]
    """
    # 检查并修复数据类型不匹配问题
    print(f"    原始潜在编码类型: {z.dtype}")
    
    # 获取解码器的参数类型
    decoder_dtype = next(decoder.parameters()).dtype
    print(f"    解码器参数类型: {decoder_dtype}")
    
    # 转换潜在编码为解码器相同的类型
    if z.dtype != decoder_dtype:
        print(f"    转换潜在编码类型: {z.dtype} -> {decoder_dtype}")
        z = z.to(dtype=decoder_dtype)
    
    # 确保潜在编码格式正确
    if z.dim() == 2:
        # 如果是2D，尝试重塑为3D
        if z.shape == (16, 4):
            z = z.view(4, 4, 4)  # [C, H, W]
        else:
            raise ValueError(f"Unexpected 2D tensor shape: {z.shape}")
    elif z.dim() == 1:
        # 如果是1D，重塑为3D
        if z.numel() == 64:  # 4*4*4
            z = z.view(4, 4, 4)
        else:
            raise ValueError(f"Unexpected 1D tensor size: {z.numel()}")
    elif z.dim() == 3:
        # 3D张量 [H, W, C] -> [C, H, W]，检查是否需要重新排列
        if z.shape == (16, 16, 4):
            z = z.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            print(f"    重新排列维度: (16,16,4) -> {z.shape}")
    elif z.dim() != 3:
        raise ValueError(f"Latent must be 1D, 2D, or 3D, got {z.dim()}D")
    
    # 添加batch维度并移到设备
    z_batch = z.unsqueeze(0).to(device=device, dtype=decoder_dtype)  # [1, C, H, W]
    
    # 解码
    with torch.no_grad():
        recon = decoder(z_batch).squeeze(0).cpu()  # [C, H, W]
    
    # 确保输出在[0,1]范围
    recon = torch.clamp(recon, 0, 1)
    
    return recon

def save_tensor_as_png(tensor: torch.Tensor, save_path: str):
    """将tensor保存为PNG图像"""
    # 确保tensor是[C, H, W]格式
    if tensor.ndim == 3:
        # 如果是多通道，取第一个通道或转为灰度
        if tensor.shape[0] > 1:
            tensor = tensor.mean(dim=0, keepdim=True)  # 转为灰度
        tensor = tensor[0]  # 去除通道维度
    
    # 转换为numpy并调整到[0, 255]
    img_array = (tensor.numpy() * 255).astype(np.uint8)
    
    # 创建PIL图像并保存
    img = Image.fromarray(img_array, mode='L')  # 灰度图像
    img.save(save_path)
    print(f"[INFO] 图像已保存到: {save_path}")

class LatentAccessor:
    def __init__(self, raw, layout: str = "style_content"):
        if isinstance(raw, dict):
            if "latents" in raw:
                raw = raw["latents"]
            elif "data" in raw:
                raw = raw["data"]
        self.raw = raw
        self.layout = layout
        
        if isinstance(raw, torch.Tensor):
            self.total_samples = raw.shape[0]
            print(f"[INFO] 总样本数: {self.total_samples}")
            print(f"[INFO] 数据形状: {raw.shape}")
            
            # 假设数据组织结构
            self.num_characters = 500
            self.num_styles_per_char = self.total_samples // self.num_characters
            print(f"[INFO] 推断结构: {self.num_characters} 字符 × {self.num_styles_per_char} 风格/字符")

    def get(self, content_i: int, style_p: int) -> torch.Tensor:
        r = self.raw
        if isinstance(r, torch.Tensor):
            if content_i >= self.num_characters:
                raise IndexError(f"content_i {content_i} >= num_characters {self.num_characters}")
            if style_p >= self.num_styles_per_char:
                raise IndexError(f"style_p {style_p} >= num_styles_per_char {self.num_styles_per_char}")
                
            idx = content_i * self.num_styles_per_char + style_p
            if idx >= self.total_samples:
                raise IndexError(f"Computed index {idx} >= total_samples {self.total_samples}")
            
            return r[idx]
        else:
            raise RuntimeError("Unsupported format")

def visualize_latents_from_pt(
    pt_file_path: str = "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/Font-Latent-Full-PT/font_latents_v2.pt",
    save_dir: str = "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0/",
    num_samples: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    从PT文件中解码潜在编码并保存为PNG图像
    
    Args:
        pt_file_path: PT文件路径
        save_dir: 保存目录
        num_samples: 要保存的样本数量
        device: 计算设备
    """
    print("🎨 开始可视化潜在编码...")
    print(f"📁 PT文件: {pt_file_path}")
    print(f"💾 保存目录: {save_dir}")
    print(f"🔢 样本数量: {num_samples}")
    print(f"🖥️  设备: {device}")
    
    # 1. 加载数据
    print("\n📥 加载潜在编码数据...")
    try:
        raw_data = load_latents_from_local(pt_file_path, map_location="cpu")
        accessor = LatentAccessor(raw_data)
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return
    
    # 2. 加载VAE解码器
    print("\n🔧 加载VAE解码器...")
    try:
        decoder, cfg, device_used = load_vae_decoder(device=device)
    except Exception as e:
        print(f"❌ 加载VAE解码器失败: {e}")
        return
    
    # 3. 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 4. 解码并保存样本
    print(f"\n🖼️  开始解码和保存 {num_samples} 个样本...")
    
    successful_saves = 0
    
    for i in range(num_samples):
        try:
            # 选择不同的内容和风格组合
            content_idx = i % min(10, accessor.num_characters)  # 循环使用前10个字符
            style_idx = (i // 10) % min(10, accessor.num_styles_per_char)  # 循环使用前10种风格
            
            print(f"  处理样本 {i+1}/{num_samples}: content={content_idx}, style={style_idx}")
            
            # 获取潜在编码
            latent = accessor.get(content_idx, style_idx)
            print(f"    潜在编码形状: {latent.shape}")
            
            # 解码为图像
            img_tensor = decode_to_image(decoder, latent, cfg, device_used)
            print(f"    解码图像形状: {img_tensor.shape}")
            
            # 保存图像
            save_path = os.path.join(save_dir, f"sample_{i+1:03d}_c{content_idx}_s{style_idx}.png")
            save_tensor_as_png(img_tensor, save_path)
            
            successful_saves += 1
            
        except Exception as e:
            print(f"    ❌ 样本 {i+1} 处理失败: {e}")
            continue
    
    print(f"\n✅ 完成! 成功保存了 {successful_saves}/{num_samples} 个图像")
    print(f"📁 图像保存在: {save_dir}")
    
    # 5. 显示数据信息总结
    print(f"\n📊 数据信息总结:")
    print(f"  - 总样本数: {accessor.total_samples}")
    print(f"  - 推断字符数: {accessor.num_characters}")
    print(f"  - 每字符风格数: {accessor.num_styles_per_char}")
    if hasattr(accessor.raw, 'shape'):
        print(f"  - 原始数据形状: {accessor.raw.shape}")

def quick_check_samples():
    """快速检查几个样本"""
    print("🔍 快速检查样本...")
    visualize_latents_from_pt(num_samples=5)

def check_more_samples():
    """检查更多样本"""
    print("🔍 检查更多样本...")
    visualize_latents_from_pt(num_samples=50)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "more":
        check_more_samples()
    else:
        print("🔍 启动快速样本检查...")
        print("💡 提示: 使用 'python decode_visualize.py more' 检查更多样本")
        quick_check_samples()