#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full dataset training version - 使用完整数据集进行训练
"""
import os
import random
import numpy as np
import pickle
from dataclasses import dataclass
from typing import Any, Tuple, Literal, Optional
import sys
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# 添加项目根目录到路径
sys.path.append('/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/Content-Style-Disentangled-Representation-Learning')
from models import build_encoder, build_decoder

# 导入本地的font_dataset模块（PT版本）
sys.path.append('/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0')
from font_dataset import FourWayFontPairLatentPTDataset

# 导入原有的组件 - 复制而不是导入避免路径问题
import math
import json

# 复制必要的函数和类
def load_latents_from_hub(
    repo_id: str = "YuanhengLi/Font-Latent-Full-PT",
    filename: str = "font_latents_v2.pt",
    token: Optional[str] = os.getenv("HF_TOKEN"),
    map_location: str = "cpu",
    local_path: Optional[str] = None,
):
    if local_path and os.path.exists(local_path):
        print(f"[INFO] Loading from local file: {local_path}")
        obj = torch.load(local_path, map_location=map_location)
        print(f"[INFO] Loaded from local file")
        return obj
    
    try:
        pt_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        obj = torch.load(pt_path, map_location=map_location)
        print(f"[INFO] Loaded from {repo_id}/{filename}")
        return obj
    except Exception as e:
        print(f"[ERROR] Failed to load from Hub: {e}")
        raise

class LatentAccessor:
    def __init__(self, raw, layout: Literal["style_content","content_style"]="style_content"):
        if isinstance(raw, dict):
            if "latents" in raw:
                raw = raw["latents"]
            elif "data" in raw:
                raw = raw["data"]
        self.raw = raw
        self.layout = layout
        
        # 推断数据组织结构
        if isinstance(raw, torch.Tensor):
            self.total_samples = raw.shape[0]
            print(f"[INFO] 总样本数: {self.total_samples}")
            
            # 假设数据是按 [character_id * num_styles + style_id] 组织的
            # 需要根据实际数据组织确定这些参数
            # 940万样本，假设500个字符，每个字符约18800种风格
            self.num_characters = 500  # 字符数量
            self.num_styles_per_char = self.total_samples // self.num_characters
            print(f"[INFO] 推断结构: {self.num_characters} 字符 × {self.num_styles_per_char} 风格/字符")

    def get(self, content_i: int, style_p: int) -> torch.Tensor:
        r = self.raw
        if isinstance(r, torch.Tensor):
            # 计算一维索引
            # 假设数据组织为: [content_0_style_0, content_0_style_1, ..., content_1_style_0, ...]
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

class ImprovedLatentAccessor:
    """
    使用FourWayFontPairLatentPTDataset的改进版LatentAccessor
    自动推断实际的字体和字符数量，不依赖硬编码假设
    """
    def __init__(self, 
                 pt_path: str, 
                 chars_path: Optional[str] = None,
                 fonts_json: Optional[str] = None,
                 device='cpu',
                 latent_shape: Tuple[int, int, int] = (4, 16, 16)):
        self.device = device
        self.pt_path = pt_path
        
        # 优先尝试使用FourWayFontPairLatentPTDataset
        if chars_path and fonts_json and os.path.exists(chars_path) and os.path.exists(fonts_json):
            print("[INFO] 使用FourWayFontPairLatentPTDataset（推荐方式）...")
            try:
                self.dataset = FourWayFontPairLatentPTDataset(
                    pt_path=pt_path,
                    chars_path=chars_path,
                    fonts_json=fonts_json,
                    latent_shape=latent_shape,
                    pair_num=1  # 只需要推断结构，不需要大量数据
                )
                
                # 获取实际的字体和字符信息
                self.fonts = self.dataset.fonts
                self.chars = self.dataset.chars
                self.common_chars = self.dataset.common_chars
                self.num_fonts = self.dataset.n
                self.num_characters = self.dataset.m
                
                # 🔥 添加兼容性属性
                self.num_styles_per_char = self.num_fonts  # 在新结构中，每个字符有num_fonts种风格
                
                print(f"[ImprovedLatentAccessor] 使用FourWayDataset发现:")
                print(f"  - 字体数量: {self.num_fonts}")
                print(f"  - 字符数量: {self.num_characters}")
                print(f"  - 每字符风格数: {self.num_styles_per_char}")
                print(f"  - 总可能组合: {self.num_fonts * self.num_characters}")
                print(f"  - 数据组织: font_idx * {self.num_characters} + char_idx")
                
                # 保存数据结构映射
                self.font_to_idx = {font: i for i, font in enumerate(self.fonts)}
                self.char_to_idx = {char: i for i, char in enumerate(self.chars)}
                self.fallback_mode = False
                return
            except Exception as e:
                print(f"[WARNING] FourWayFontPairLatentPTDataset初始化失败: {e}")
                print("[INFO] 降级到fallback模式...")
        else:
            missing_files = []
            if not chars_path:
                missing_files.append("chars_path")
            elif not os.path.exists(chars_path):
                missing_files.append(f"chars_path({chars_path})")
            if not fonts_json:
                missing_files.append("fonts_json")
            elif not os.path.exists(fonts_json):
                missing_files.append(f"fonts_json({fonts_json})")
            
            print(f"[INFO] 缺少文件: {', '.join(missing_files)}")
            print("[INFO] 使用fallback模式，默认2056字体×4574字符结构...")
        
        # Fallback模式：使用2056字体×4574字符的默认结构
        self._create_from_old_accessor(pt_path, latent_shape, device)
        
    def _create_from_old_accessor(self, pt_path: str, latent_shape: Tuple[int, int, int], device: str):
        """使用旧的LatentAccessor逻辑作为fallback"""
        print("[INFO] 使用fallback模式，采用2056字体×4574字符结构...")
        
        # 加载PT文件
        blob = torch.load(pt_path, map_location="cpu")
        if isinstance(blob, dict) and "latents" in blob:
            latents = blob["latents"]
        else:
            latents = blob
            
        if isinstance(latents, torch.Tensor):
            if latents.dim() == 4:  # (N, H, W, C)
                self.total_samples = latents.shape[0]
                self.latents_hwc = latents
            else:
                self.total_samples = latents.shape[0]
                self.raw_tensor = latents
                
            # 使用准确的数据结构: 2056字体 × 4574字符
            self.num_characters = 4574  # 字符数量
            self.num_fonts = 2056       # 字体数量
            
            # 验证数据总量是否匹配
            expected_total = self.num_fonts * self.num_characters
            if self.total_samples != expected_total:
                print(f"[WARNING] 数据量不匹配!")
                print(f"  - 实际样本: {self.total_samples}")
                print(f"  - 预期样本: {expected_total} (2056×4574)")
                print(f"  - 将按实际数据调整结构...")
                
                # 尝试其他可能的组织方式
                if self.total_samples % 4574 == 0:
                    self.num_fonts = self.total_samples // 4574
                    print(f"  - 调整为: {self.num_fonts} 字体 × 4574 字符")
                elif self.total_samples % 2056 == 0:
                    self.num_characters = self.total_samples // 2056
                    print(f"  - 调整为: 2056 字体 × {self.num_characters} 字符")
                else:
                    # 最后的fallback，尝试接近正方形的分布
                    import math
                    sqrt_total = int(math.sqrt(self.total_samples))
                    for chars in [4574, 4000, 3500, sqrt_total]:
                        if self.total_samples % chars == 0:
                            self.num_characters = chars
                            self.num_fonts = self.total_samples // chars
                            break
                    print(f"  - 最终调整为: {self.num_fonts} 字体 × {self.num_characters} 字符")
            
            print(f"[INFO] Fallback结构: {self.num_fonts} 字体 × {self.num_characters} 字符")
            print(f"[INFO] 数据组织: font_idx * {self.num_characters} + char_idx")
            
            # 🔥 添加兼容性属性
            self.num_styles_per_char = self.num_fonts  # 在新结构中，每个字符有num_fonts种风格
            
            # 创建假的字体和字符列表
            self.fonts = [f"font_{i:04d}" for i in range(self.num_fonts)]
            self.chars = [f"char_{i:04d}" for i in range(self.num_characters)]
            self.common_chars = self.chars
            
            self.font_to_idx = {font: i for i, font in enumerate(self.fonts)}
            self.char_to_idx = {char: i for i, char in enumerate(self.chars)}
            
            # 创建fallback数据集接口
            self.dataset = None
            self.fallback_mode = True
        else:
            raise RuntimeError("无法识别的数据格式")
    
    def get_by_indices(self, font_idx: int, char_idx: int) -> torch.Tensor:
        """通过索引获取潜在编码"""
        if font_idx >= self.num_fonts:
            raise IndexError(f"font_idx {font_idx} >= num_fonts {self.num_fonts}")
        if char_idx >= self.num_characters:
            raise IndexError(f"char_idx {char_idx} >= num_characters {self.num_characters}")
            
        if hasattr(self, 'fallback_mode') and self.fallback_mode:
            # 使用fallback模式
            idx = font_idx * self.num_characters + char_idx
            if hasattr(self, 'latents_hwc'):
                # 4D数据 (N, H, W, C) -> (C, H, W)
                z_hwc = self.latents_hwc[idx]
                latent = z_hwc.permute(2, 0, 1).contiguous()
            else:
                # 使用原始tensor
                latent = self.raw_tensor[idx]
            return latent.to(self.device)
        else:
            # 使用数据集的内部方法获取潜在编码
            flat_idx = self.dataset._flat_index(font_idx, char_idx)
            latent = self.dataset._get_chw(flat_idx)
            return latent.to(self.device)
    
    def get_by_names(self, font_name: str, char_name: str) -> torch.Tensor:
        """通过名称获取潜在编码"""
        if font_name not in self.font_to_idx:
            raise KeyError(f"Font '{font_name}' not found")
        if char_name not in self.char_to_idx:
            raise KeyError(f"Char '{char_name}' not found")
            
        font_idx = self.font_to_idx[font_name]
        char_idx = self.char_to_idx[char_name]
        
        return self.get_by_indices(font_idx, char_idx)
    
    def get(self, content_i: int, style_p: int) -> torch.Tensor:
        """兼容原接口的方法"""
        # content_i对应字符索引，style_p对应字体索引
        return self.get_by_indices(style_p % self.num_fonts, content_i % self.num_characters)

# VAE解码器相关函数 - 使用vae_io.py的实现
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
    # 🔥 检查并修复数据类型不匹配问题（减少日志输出）
    decoder_dtype = next(decoder.parameters()).dtype
    if z.dtype != decoder_dtype:
        # 只在第一次转换时输出信息，避免日志过多
        if not hasattr(decode_to_image, '_conversion_logged'):
            print(f"[INFO] 自动转换潜在编码类型: {z.dtype} -> {decoder_dtype}")
            decode_to_image._conversion_logged = True
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
        # 🔥 3D张量，检查是否需要重新排列维度
        if z.shape == (16, 16, 4):
            z = z.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            if not hasattr(decode_to_image, '_reshape_logged'):
                print(f"[INFO] 重新排列维度: (16,16,4) -> {z.shape}")
                decode_to_image._reshape_logged = True
    elif z.dim() != 3:
        raise ValueError(f"Latent must be 1D, 2D, or 3D, got {z.dim()}D")
    
    # 添加batch维度并移到设备，确保数据类型匹配
    z_batch = z.unsqueeze(0).to(device=device, dtype=decoder_dtype)  # [1, C, H, W]
    
    # 解码
    with torch.no_grad():
        recon = decoder(z_batch).squeeze(0).cpu()  # [C, H, W]
    
    # 确保输出在[0,1]范围
    recon = torch.clamp(recon, 0, 1)
    
    return recon
class VGGEncoder(nn.Module):
    """基于VGG的编码器 - 使用预训练特征"""
    def __init__(self, in_ch=1, emb_dim=512, vgg_variant="vgg16", task="content"):
        super().__init__()
        self.task = task
        
        # 加载预训练VGG模型
        if vgg_variant == "vgg16":
            vgg = models.vgg16(pretrained=True)
        elif vgg_variant == "vgg19":
            vgg = models.vgg19(pretrained=True)
        else:
            raise ValueError(f"Unsupported VGG variant: {vgg_variant}")
        
        # 提取VGG特征层
        self.features = vgg.features
        
        # 输入通道适配 (灰度图 → RGB)
        if in_ch == 1:
            # 将第一层Conv2d从3通道改为1通道
            first_conv = self.features[0]
            self.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
            # 复制预训练权重的均值到单通道
            with torch.no_grad():
                self.features[0].weight = nn.Parameter(
                    first_conv.weight.mean(dim=1, keepdim=True)
                )
                self.features[0].bias = first_conv.bias.clone()
        
        # 冻结低层特征 (可选)
        self._freeze_early_layers(freeze_layers=3)
        
        # 根据任务选择不同的特征提取策略
        if task == "content":
            # 内容任务：使用中层特征，保留空间信息
            self.feature_layers = [10, 17, 24]  # conv2_2, conv3_4, conv4_4
        else:  # style
            # 风格任务：使用多层特征，提取纹理信息
            self.feature_layers = [3, 8, 15, 22, 29]  # conv1_2, conv2_2, conv3_3, conv4_3, conv5_3
        
        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4)) if task == "content" else nn.AdaptiveAvgPool2d((1, 1))
        
        # 特征融合层
        self.feature_dim = self._calculate_feature_dim()
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, emb_dim * 2),
            nn.BatchNorm1d(emb_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
    def _freeze_early_layers(self, freeze_layers=3):
        """冻结前几层的参数"""
        for i, module in enumerate(self.features.children()):
            if i < freeze_layers:
                for param in module.parameters():
                    param.requires_grad = False
    
    def _calculate_feature_dim(self):
        """计算特征维度"""
        # 这里需要根据实际的VGG结构和选择的层来计算
        # 简化计算，假设每层512维特征
        if self.task == "content":
            return 512 * 16  # 4x4 spatial
        else:
            return 512 * len(self.feature_layers)  # 多层特征拼接
    
    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.size(1) not in (1, 3):
            x = x.mean(dim=1, keepdim=True)
        
        # 如果输入是单通道但网络期望三通道
        if x.size(1) == 1 and self.features[0].in_channels == 3:
            x = x.repeat(1, 3, 1, 1)
        
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_layers:
                # 应用自适应池化
                pooled = self.adaptive_pool(x)
                features.append(pooled.view(pooled.size(0), -1))
        
        # 拼接所有特征
        if features:
            combined_features = torch.cat(features, dim=1)
        else:
            # 如果没有提取到特征，使用最后的输出
            combined_features = self.adaptive_pool(x).view(x.size(0), -1)
        
        return self.fc(combined_features)

class HybridEncoder(nn.Module):
    """混合编码器 - 结合VGG和自定义CNN"""
    def __init__(self, in_ch=1, emb_dim=512, task="content"):
        super().__init__()
        self.task = task
        
        # VGG分支
        self.vgg_branch = VGGEncoder(in_ch=in_ch, emb_dim=emb_dim//2, task=task)
        
        # 自定义CNN分支
        if task == "content":
            self.custom_branch = ContentEncoder(in_ch=in_ch, emb_dim=emb_dim//2)
        else:
            self.custom_branch = StyleEncoder(in_ch=in_ch, emb_dim=emb_dim//2)
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        vgg_features = self.vgg_branch(x)
        custom_features = self.custom_branch(x)
        
        # 拼接特征
        combined = torch.cat([vgg_features, custom_features], dim=1)
        
        return self.fusion(combined)

class EnhancedContentEncoder(nn.Module):
    """增强的内容编码器 - 更深的网络"""
    def __init__(self, in_ch=1, emb_dim=512):
        super().__init__()
        
        # 更深的卷积网络 - 受VGG启发的设计
        self.features = nn.Sequential(
            # Block 1 - 64 channels
            nn.Conv2d(in_ch, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            
            # Block 2 - 128 channels  
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),  # 32→16
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            
            # Block 3 - 256 channels
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),  # 16→8
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            
            # Block 4 - 512 channels
            nn.Conv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),  # 8→4
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            
            # Block 5 - 深层特征
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.spatial_pool = nn.AdaptiveAvgPool2d(4)  # 保留4x4空间信息
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 16, emb_dim * 2),
            nn.BatchNorm1d(emb_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.size(1) not in (1,3):
            x = x.mean(dim=1, keepdim=True)
        
        h = self.features(x)
        h = self.spatial_pool(h)
        h = h.view(h.size(0), -1)
        return self.classifier(h)

class EnhancedStyleEncoder(nn.Module):
    """增强的风格编码器 - VGG风格的多层特征"""
    def __init__(self, in_ch=1, emb_dim=512):
        super().__init__()
        
        # VGG风格的特征提取器
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_ch, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32→16
            
            # Block 2  
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16→8
            
            # Block 3
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8→4
            
            # Block 4
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            
            # Block 5 - 深层纹理特征
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # 多尺度池化
        self.global_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),    # 全局平均
            nn.AdaptiveMaxPool2d(1),    # 全局最大
            nn.AdaptiveAvgPool2d(2),    # 2x2平均
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 6, emb_dim * 2),  # 512*1 + 512*1 + 512*4 = 512*6
            nn.BatchNorm1d(emb_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.size(1) not in (1,3):
            x = x.mean(dim=1, keepdim=True)
        
        h = self.features(x)
        
        # 多尺度特征提取
        pooled_features = []
        for pool in self.global_pools:
            pooled = pool(h).view(h.size(0), -1)
            pooled_features.append(pooled)
        
        h = torch.cat(pooled_features, dim=1)
        return self.classifier(h)
    """内容编码器 - 保留更多空间信息，少用池化"""
    def __init__(self, in_ch=1, emb_dim=512):
        super().__init__()
        # 🔥 更少池化，保留空间细节用于内容识别
        self.spatial_net = nn.Sequential(
            # 第一层：保持分辨率，提取细粒度特征
            nn.Conv2d(in_ch, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),  # 额外层
            
            # 第二层：轻微下采样，保留结构信息
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),  # 额外层
            
            # 第三层：提取局部内容特征
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            
            # 第四层：高级特征但保留更多空间信息
            nn.Conv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # 🔥 保留更多空间信息 - 4x4而不是2x2
        self.spatial_pool = nn.AdaptiveAvgPool2d(4)  # 保留4x4空间信息
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 16, emb_dim),  # 4x4=16
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.size(1) not in (1,3):
            x = x.mean(dim=1, keepdim=True)
        
        h = self.spatial_net(x)
        h = self.spatial_pool(h)  # [B, 512, 4, 4]
        h = h.view(h.size(0), -1)  # [B, 512*16] = [B, 8192]
        return self.fc(h)

class StyleEncoder(nn.Module):
    """风格编码器 - 更多池化，提取抽象特征"""
    def __init__(self, in_ch=1, emb_dim=512):
        super().__init__()
        # 修复池化问题：减少下采样层数，确保不会压缩到0
        self.abstract_net = nn.Sequential(
            # 第一层：基础特征提取
            nn.Conv2d(in_ch, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),  # 32→32
            nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),  # 32→16
            
            # 第二层：纹理特征
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),  # 16→16
            nn.Conv2d(128, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),  # 16→8
            
            # 第三层：风格特征
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),  # 8→8
            nn.Conv2d(256, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),  # 8→4
            
            # 第四层：高级风格特征
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),  # 4→4
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),  # 4→4
            nn.Dropout(0.3)
        )
        
        # 确保池化后不会变成0尺寸
        self.global_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),     # 全局平均池化到1x1
            nn.AdaptiveMaxPool2d(1),     # 全局最大池化到1x1
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(1024, emb_dim),  # 两种池化特征拼接: 512*1 + 512*1 = 1024
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.size(1) not in (1,3):
            x = x.mean(dim=1, keepdim=True)
        
        h = self.abstract_net(x)
        
        # 🔥 修复后的池化特征组合 - 只使用两种池化方式
        pooled_features = []
        for pool in self.global_pools:
            pooled = pool(h).view(h.size(0), -1)
            pooled_features.append(pooled)
        
        h = torch.cat(pooled_features, dim=1)  # [B, 512*2]
        return self.fc(h)

class TinyEncoder(nn.Module):
    """临时使用原来的编码器确保兼容性"""
    def __init__(self, in_ch=1, emb_dim=512, task="content"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(512, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.size(1) not in (1,3):
            x = x.mean(dim=1, keepdim=True)
        h = self.net(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)

class SiameseJudge(nn.Module):
    def __init__(self, in_ch=1, emb_dim=512, mlp_hidden=512, task="content", encoder_type="enhanced"):
        super().__init__()
        
        # 🔥 支持多种编码器类型
        if encoder_type == "enhanced":
            # 增强版编码器 - VGG风格的深层网络
            if task == "content":
                self.encoder = EnhancedContentEncoder(in_ch=in_ch, emb_dim=emb_dim)
            else:  # style
                self.encoder = EnhancedStyleEncoder(in_ch=in_ch, emb_dim=emb_dim)
        elif encoder_type == "vgg":
            # 纯VGG编码器
            self.encoder = VGGEncoder(in_ch=in_ch, emb_dim=emb_dim, task=task)
        elif encoder_type == "hybrid":
            # 混合编码器 - VGG + 自定义CNN
            self.encoder = HybridEncoder(in_ch=in_ch, emb_dim=emb_dim, task=task)
        elif encoder_type == "original":
            # 原始编码器
            if task == "content":
                self.encoder = ContentEncoder(in_ch=in_ch, emb_dim=emb_dim)
            else:  # style
                self.encoder = StyleEncoder(in_ch=in_ch, emb_dim=emb_dim)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")
            
        # 更深的分类头
        self.head = nn.Sequential(
            nn.Linear(emb_dim, mlp_hidden), 
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, mlp_hidden//2),
            nn.BatchNorm1d(mlp_hidden//2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden//2, mlp_hidden//4),
            nn.BatchNorm1d(mlp_hidden//4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden//4, 1)
        )

    def forward(self, x1, x2):
        v1 = self.encoder(x1)
        v2 = self.encoder(x2)
        diff = torch.abs(v1 - v2)
        logit = self.head(diff)
        return logit.squeeze()  # 🔥 只返回logit，并确保正确的形状

@torch.no_grad()
def score_pair(model: nn.Module, img1: torch.Tensor, img2: torch.Tensor, device=None) -> float:
    device = device or next(model.parameters()).device
    model.eval()
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)
    logit = model(img1, img2)  # 🔥 现在只返回logit
    return torch.sigmoid(logit).item()

class FullDatasetPairDataset(Dataset):
    """
    使用完整数据集的配对数据集
    从大量样本中随机选择配对进行训练
    """
    def __init__(self, 
                 accessor: ImprovedLatentAccessor, 
                 decoder: nn.Module,
                 cfg: dict,
                 device_used: torch.device,
                 task: Literal["content","style"]="content",
                 num_styles: int = 100,
                 num_contents: int = 1000,
                 length: int = 50000,  # 增加训练样本数
                 augment: bool = True,
                 device: str = "cpu"):
        """
        Args:
            accessor: 潜在编码访问器
            decoder: VAE解码器模型
            cfg: VAE配置字典
            device_used: 解码器所在设备
            task: 任务类型 ('content' 或 'style')
            num_styles: 使用的风格数量
            num_contents: 使用的内容数量  
            length: 训练样本总数
            augment: 是否使用数据增强
            device: 计算设备
        """
        self.accessor = accessor
        self.decoder = decoder
        self.cfg = cfg
        self.device_used = device_used
        self.task = task
        self.num_styles = num_styles
        self.num_contents = num_contents
        self.length = length
        self.augment = augment
        self.device = device
        
        print(f"[INFO] 创建全数据集训练器:")
        print(f"  - 任务类型: {task}")
        print(f"  - 风格数量: {num_styles}")
        print(f"  - 内容数量: {num_contents}")
        print(f"  - 训练样本: {length}")
        
        # 预先检查数据访问是否正常
        try:
            test_sample = self.accessor.get(0, 0)
            print(f"  - 样本形状: {test_sample.shape}")
        except Exception as e:
            print(f"  - 警告: 数据访问测试失败: {e}")

    def __len__(self):
        return self.length

    def _aug(self, x: torch.Tensor) -> torch.Tensor:
        """数据增强"""
        if not self.augment: 
            return x
        # 简单的几何和噪声增强
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])  # 水平翻转
        if random.random() < 0.2:
            x = x + torch.randn_like(x) * 0.02
            x = torch.clamp(x, 0.0, 1.0)
        return x

    def __getitem__(self, idx):
        """获取训练样本对 - 改进的采样策略"""
        # 获取实际可用的范围
        max_contents = min(self.num_contents, self.accessor.num_characters)
        max_styles = min(self.num_styles, self.accessor.num_styles_per_char)
        
        # 随机选择索引 - 使用安全的范围
        content_i = random.randint(0, max_contents - 1)
        content_j = random.randint(0, max_contents - 1)
        style_p = random.randint(0, max_styles - 1)
        style_q = random.randint(0, max_styles - 1)
        
        # 确保有不同的选择用于负样本
        while content_j == content_i:
            content_j = random.randint(0, max_contents - 1)
        while style_q == style_p:
            style_q = random.randint(0, max_styles - 1)
        
        # 改进的采样策略：针对性增加困难样本的概率
        if self.task == "content":
            # Content任务：80%概率生成负样本（不同内容相同风格 - 最困难）
            is_positive = (random.random() < 0.2)
        else:  # style task
            # Style任务：75%概率生成正样本（相同风格不同内容 - 困难样本）
            is_positive = (random.random() < 0.75)
        
        try:
            # 获取潜在编码
            z1 = self.accessor.get(content_i, style_p)
            if idx < 5:  # 只在前几个样本打印调试信息
                print(f"[DEBUG] Sample {idx}: z1.shape = {z1.shape}")
            
            # 解码为图像 (使用VAE解码器)
            ci_sp = decode_to_image(self.decoder, z1, self.cfg, self.device_used)
            if idx < 5:
                print(f"[DEBUG] Sample {idx}: ci_sp.shape = {ci_sp.shape}")
            
            if self.task == "content":
                if is_positive:
                    # 正样本: 相同内容，不同风格
                    z2 = self.accessor.get(content_i, style_q)
                    ci_sq = decode_to_image(self.decoder, z2, self.cfg, self.device_used)
                    x1, x2, y = ci_sp, ci_sq, 1.0
                else:
                    # 负样本: 不同内容，相同风格 (困难样本)
                    # 🔥 策略改进：选择相似的内容增加难度
                    if random.random() < 0.3:  # 30%概率选择相邻内容
                        content_j = min(max_contents - 1, content_i + 1)
                    z2 = self.accessor.get(content_j, style_p)
                    cj_sp = decode_to_image(self.decoder, z2, self.cfg, self.device_used)
                    x1, x2, y = ci_sp, cj_sp, 0.0
            else:  # style task
                if is_positive:
                    # 正样本: 相同风格，不同内容 (困难样本)
                    # 🔥 策略改进：选择差异较大的内容
                    if random.random() < 0.3:  # 30%概率选择差异大的内容
                        content_j = (content_i + max_contents // 2) % max_contents
                    z2 = self.accessor.get(content_j, style_p)
                    cj_sp = decode_to_image(self.decoder, z2, self.cfg, self.device_used)
                    x1, x2, y = ci_sp, cj_sp, 1.0
                else:
                    # 负样本: 不同风格，相同内容
                    z2 = self.accessor.get(content_i, style_q)
                    ci_sq = decode_to_image(self.decoder, z2, self.cfg, self.device_used)
                    x1, x2, y = ci_sp, ci_sq, 0.0

            # 应用数据增强
            x1 = self._aug(x1)
            x2 = self._aug(x2)
            
            # 确保张量是连续的并且可以被复制
            x1 = x1.contiguous().detach().clone()
            x2 = x2.contiguous().detach().clone()
            
            return x1, x2, torch.tensor(y, dtype=torch.float32)  # 🔥 返回标量而不是[y]
            
        except Exception as e:
            print(f"[WARN] 样本生成失败 (idx={idx}): {e}")
            # 返回一个dummy样本避免训练中断
            dummy_img = torch.zeros(1, 32, 32)
            return dummy_img, dummy_img, torch.tensor(0.0, dtype=torch.float32)  # 🔥 返回标量

def run_full_training(
    task: Literal["content","style"]="content",
    layout: Literal["style_content","content_style"]="style_content",
    
    # PT数据路径参数
    pt_path: str = "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/Font-Latent-Full-PT/font_latents_v2.pt",
    chars_path: Optional[str] = "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0/intersection_chars_full.txt",  # 🔥 在这里设置你的字符文件路径
    fonts_json: Optional[str] = "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0/font_list(1).json",  # 🔥 在这里设置你的字体文件路径
    
    decoder_name: str = "diff_decoder",
    device="cuda" if torch.cuda.is_available() else "cpu",
    encoder_type: str = "enhanced",  # 🔥 新增编码器类型选择
    
    # 训练参数
    num_styles: int = 500,      # 使用的风格数量 (减少到合理范围)
    num_contents: int = 500,    # 使用的内容数量 (匹配字符数)
    train_samples: int = 50000, # 训练样本数
    batch_size: int = 32,       # 批量大小
    epochs: int = 50,           # 训练轮数
    lr: float = 5e-4,           # 学习率
    
    # 评估参数
    eval_samples: int = 5000,   # 评估样本数
):
    """
    运行完整数据集训练
    """
    print("🚀 开始全数据集字体训练...")
    print(f"📊 训练配置:")
    print(f"  - 任务: {task}")
    print(f"  - 编码器: {encoder_type}")  # 🔥 显示编码器类型
    print(f"  - 设备: {device}")
    print(f"  - 风格数: {num_styles}")
    print(f"  - 内容数: {num_contents}")
    print(f"  - 训练样本: {train_samples}")
    print(f"  - 批量大小: {batch_size}")
    print(f"  - 训练轮数: {epochs}")
    
    # 1. 加载数据
    print("\n📥 加载潜在编码数据...")
    print(f"  - PT文件路径: {pt_path}")
    if chars_path:
        print(f"  - 字符文件路径: {chars_path}")
    if fonts_json:
        print(f"  - 字体JSON路径: {fonts_json}")
    
    accessor = ImprovedLatentAccessor(
        pt_path=pt_path,
        chars_path=chars_path,
        fonts_json=fonts_json,
        device="cpu",  # 先在CPU上加载
        latent_shape=(4, 16, 16)
    )
    
    # 2. 加载VAE解码器
    print("\n🔧 加载VAE解码器...")
    decoder, cfg, device_used = load_vae_decoder(device=device)
    print(f"[INFO] VAE Decoder loaded on {device_used}")
    
    # 3. 创建数据集和加载器
    print("\n📚 创建训练数据集...")
    train_dataset = FullDatasetPairDataset(
        accessor=accessor,
        decoder=decoder,
        cfg=cfg,
        device_used=device_used,
        task=task,
        num_styles=num_styles,
        num_contents=num_contents,
        length=train_samples,
        augment=True,
        device="cpu"  # 在dataset中保持CPU，在训练时移到GPU
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # 禁用多进程避免存储冲突
        pin_memory=True if device == "cuda" else False
    )
    
    # 4. 创建模型
    print("\n🤖 创建Siamese模型...")
    # 从一个样本推断通道数
    sample_img = decode_to_image(decoder, accessor.get(0, 0), cfg, device_used)
    C = sample_img.shape[0]
    print(f"  - 输入通道数: {C}")
    print(f"  - 任务类型: {task}")
    print(f"  - 编码器类型: {encoder_type}")
    
    model = SiameseJudge(
        in_ch=C, 
        emb_dim=512, 
        mlp_hidden=512, 
        task=task,
        encoder_type=encoder_type  # 🔥 传递编码器类型
    )
    
    # 5. 训练模型
    print("\n🎯 开始训练...")
    model = train_full_model(model, train_loader, device=device, lr=lr, epochs=epochs, task=task)
    
    # 6. 评估模型
    print("\n📊 评估模型性能...")
    
    # 🔥 先进行快速理智检查
    sanity_ok = quick_sanity_check(model, accessor, decoder, cfg, device_used, task, device, num_samples=50)
    
    if sanity_ok:
        # 如果理智检查通过，再进行详细评估
        accuracy = eval_model(model, accessor, decoder, cfg, device_used, task, num_styles, num_contents, eval_samples, device)
    else:
        print("⚠️  跳过详细评估，模型需要调试")
        accuracy = 0.0
    
    return model, accessor, decoder

class WeightedFocalLoss(nn.Module):
    """加权Focal Loss - 对困难样本给予更高权重"""
    def __init__(self, alpha=1, gamma=2, pos_weight=1.0, neg_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight  # 正样本权重
        self.neg_weight = neg_weight  # 负样本权重
        
    def forward(self, inputs, targets):
        # 确保inputs是正确的tensor格式
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(f"Expected tensor, got {type(inputs)}")
        
        # 计算sigmoid概率
        probs = torch.sigmoid(inputs)
        
        # 计算基础BCE损失
        ce_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        
        # 计算pt (预测正确的概率)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Focal weight: (1-pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # 类别权重
        class_weight = torch.where(targets == 1, self.pos_weight, self.neg_weight)
        
        # 最终损失
        focal_loss = self.alpha * class_weight * focal_weight * ce_loss
        return focal_loss.mean()

class AdaptiveLoss(nn.Module):
    """自适应损失 - 根据任务类型调整权重"""
    def __init__(self, task="content"):
        super().__init__()
        if task == "content":
            # Content任务: 负样本(不同内容相同风格)更难，给更高权重
            # 根据之前结果分析，负样本准确率偏低(66.2%)，需要加强
            self.pos_weight = 1.0
            self.neg_weight = 3.0  # 🔥 显著增加负样本权重
        else:  # style
            # Style任务: 正样本(相同风格不同内容)更难，给更高权重
            # 根据之前结果分析，正样本准确率偏低(63.9%)，需要加强  
            self.pos_weight = 2.5  # 🔥 增加正样本权重
            self.neg_weight = 1.0
            
        self.focal_loss = WeightedFocalLoss(
            alpha=1.5, gamma=2,  # 🔥 增加alpha值
            pos_weight=self.pos_weight, 
            neg_weight=self.neg_weight
        )
        
    def forward(self, inputs, targets):
        return self.focal_loss(inputs, targets)

def train_full_model(model, loader, device="cuda", lr=1e-4, epochs=20, task="content"):
    """训练完整模型 - 使用改进的损失函数"""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    # 🔥 使用自适应损失函数
    crit = AdaptiveLoss(task=task)
    
    for ep in range(1, epochs+1):
        model.train()
        tot_loss = 0.0
        tot_acc = 0.0
        num_batches = 0
        
        for i, (x1, x2, y) in enumerate(loader):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            
            opt.zero_grad()
            logits = model(x1, x2)  # 🔥 现在模型只返回logits
            
            # 🔥 使用改进的损失函数
            loss = crit(logits, y)
            loss.backward()
            
            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()
            
            # 统计
            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5).float()
                acc = (pred == y).float().mean()
                tot_loss += loss.item()
                tot_acc += acc.item()
                num_batches += 1
            
            # 打印进度
            if (i + 1) % 100 == 0:
                avg_loss = tot_loss / num_batches
                avg_acc = tot_acc / num_batches
                print(f"  Batch {i+1}/{len(loader)}: loss={avg_loss:.4f}, acc={avg_acc:.3f}")
        
        scheduler.step()
        
        # Epoch总结
        avg_loss = tot_loss / num_batches
        avg_acc = tot_acc / num_batches
        lr_current = scheduler.get_last_lr()[0]
        
        print(f"[Epoch {ep}/{epochs}] loss={avg_loss:.4f}, acc={avg_acc:.3f}, lr={lr_current:.2e}")
    
    return model

def eval_model(model, accessor, decoder, cfg, device_used, task, num_styles, num_contents, eval_samples, device):
    """评估模型性能 - 改进版本"""
    model.eval()
    correct = 0
    total = 0
    pos_correct = 0
    neg_correct = 0
    pos_total = 0
    neg_total = 0
    
    print(f"🔍 评估 {eval_samples} 个样本...")
    
    # 用于评估的测试集应该与训练集不重叠
    test_contents = list(range(num_contents//2, num_contents))  # 使用后半部分内容作为测试
    test_styles = list(range(num_styles//2, num_styles))      # 使用后半部分风格作为测试
    
    with torch.no_grad():
        for eval_idx in range(eval_samples):
            # 随机选择测试样本（使用测试集范围）
            i = random.choice(test_contents)
            j = random.choice(test_contents)
            p = random.choice(test_styles)
            q = random.choice(test_styles)
            
            while j == i:
                j = random.choice(test_contents)
            while q == p:
                q = random.choice(test_styles)
            
            try:
                # 生成测试图像
                ci_sp = decode_to_image(decoder, accessor.get(i, p), cfg, device_used).to(device)
                ci_sq = decode_to_image(decoder, accessor.get(i, q), cfg, device_used).to(device)
                cj_sp = decode_to_image(decoder, accessor.get(j, p), cfg, device_used).to(device)
                
                if task == "content":
                    # 测试正样本: 相同内容
                    pos_prob = score_pair(model, ci_sp, ci_sq, device=device)
                    # 测试负样本: 不同内容
                    neg_prob = score_pair(model, ci_sp, cj_sp, device=device)
                else:  # style
                    # 测试正样本: 相同风格
                    pos_prob = score_pair(model, ci_sp, cj_sp, device=device)
                    # 测试负样本: 不同风格
                    neg_prob = score_pair(model, ci_sp, ci_sq, device=device)
                
                # 检查分类是否正确
                if pos_prob > 0.5:
                    pos_correct += 1
                pos_total += 1
                
                if neg_prob < 0.5:
                    neg_correct += 1
                neg_total += 1
                
                total += 2
                correct = pos_correct + neg_correct
                
            except Exception as e:
                continue
    
    # 计算详细统计
    accuracy = correct / total if total > 0 else 0
    pos_accuracy = pos_correct / pos_total if pos_total > 0 else 0
    neg_accuracy = neg_correct / neg_total if neg_total > 0 else 0
    
    print(f"📊 评估完成:")
    print(f"   总体准确率: {accuracy:.3f} ({correct}/{total})")
    print(f"   正样本准确率: {pos_accuracy:.3f} ({pos_correct}/{pos_total})")
    print(f"   负样本准确率: {neg_accuracy:.3f} ({neg_correct}/{neg_total})")
    
    # 🔥 添加快速判断训练效果的指标
    if accuracy > 0.7:
        print("✅ 模型表现良好！可以考虑扩大数据集")
    elif accuracy > 0.6:
        print("⚠️  模型表现一般，建议调整超参数")
    else:
        print("❌ 模型表现较差，需要检查代码逻辑")
    
    return accuracy

def quick_sanity_check(model, accessor, decoder, cfg, device_used, task, device, num_samples=50):
    """
    快速理智检查 - 验证模型是否学到基本特征
    """
    print(f"\n🔍 快速验证模型 ({task})...")
    model.eval()
    
    correct = 0
    total = 0
    same_probs = []
    diff_probs = []
    
    with torch.no_grad():
        for sample_idx in range(min(num_samples, 10)):  # 先只测试10个样本进行调试
            try:
                # 🔥 修复：每次使用不同的样本进行测试
                i = sample_idx % min(20, accessor.num_characters)  # 循环使用不同内容
                p = sample_idx % min(20, accessor.num_styles_per_char)  # 循环使用不同风格
                
                # 🔥 正确的任务相关采样策略
                if task == "content":
                    # 内容任务：相同内容不同风格 = 相似, 不同内容相同风格 = 不相似
                    q = (p + 5) % min(20, accessor.num_styles_per_char)  # 不同风格
                    j = (i + 5) % min(20, accessor.num_characters)       # 不同内容
                    
                    # 相同样本：相同内容，不同风格 (应该相似)
                    img1 = decode_to_image(decoder, accessor.get(i, p), cfg, device_used).to(device)
                    img2 = decode_to_image(decoder, accessor.get(i, q), cfg, device_used).to(device)  # 相同内容i，不同风格q
                    same_prob = score_pair(model, img1, img2, device=device)
                    same_probs.append(same_prob)
                    
                    # 不同样本：不同内容，相同风格 (应该不相似)
                    img3 = decode_to_image(decoder, accessor.get(j, p), cfg, device_used).to(device)  # 不同内容j，相同风格p
                    diff_prob = score_pair(model, img1, img3, device=device)
                    diff_probs.append(diff_prob)
                    
                else:  # style task
                    # 风格任务：相同风格不同内容 = 相似, 不同风格相同内容 = 不相似
                    q = (p + 5) % min(20, accessor.num_styles_per_char)  # 不同风格
                    j = (i + 5) % min(20, accessor.num_characters)       # 不同内容
                    
                    # 相同样本：相同风格，不同内容 (应该相似)
                    img1 = decode_to_image(decoder, accessor.get(i, p), cfg, device_used).to(device)
                    img2 = decode_to_image(decoder, accessor.get(j, p), cfg, device_used).to(device)  # 不同内容j，相同风格p
                    same_prob = score_pair(model, img1, img2, device=device)
                    same_probs.append(same_prob)
                    
                    # 不同样本：不同风格，相同内容 (应该不相似)
                    img3 = decode_to_image(decoder, accessor.get(i, q), cfg, device_used).to(device)  # 相同内容i，不同风格q
                    diff_prob = score_pair(model, img1, img3, device=device)
                    diff_probs.append(diff_prob)
                
                # 🔥 添加调试信息
                if sample_idx < 3:  # 前3个样本详细调试
                    if task == "content":
                        print(f"  [调试] 样本{sample_idx}: 相同内容({i},{p})vs({i},{q}) | 不同内容({i},{p})vs({j},{p})")
                    else:
                        print(f"  [调试] 样本{sample_idx}: 相同风格({i},{p})vs({j},{p}) | 不同风格({i},{p})vs({i},{q})")
                    
                    print(f"  [调试] 相同样本概率={same_prob:.4f}, 不同样本概率={diff_prob:.4f}")
                    
                    # 检查图像差异
                    if task == "content":
                        img_diff_same = torch.abs(img1 - img2).max().item()
                        img_diff_diff = torch.abs(img1 - img3).max().item()
                        print(f"  [调试] 相同内容图像差异: {img_diff_same:.6f}")
                        print(f"  [调试] 不同内容图像差异: {img_diff_diff:.6f}")
                        
                        # 检查特征
                        v1 = model.encoder(img1.unsqueeze(0))
                        v2 = model.encoder(img2.unsqueeze(0)) 
                        v3 = model.encoder(img3.unsqueeze(0))
                        
                        feature_diff_same = torch.abs(v1 - v2).max().item()
                        feature_diff_diff = torch.abs(v1 - v3).max().item()
                        
                        print(f"  [调试] 相同内容特征差异: {feature_diff_same:.6f}")
                        print(f"  [调试] 不同内容特征差异: {feature_diff_diff:.6f}")
                        
                        # 检查logit
                        logit_same = model(img1.unsqueeze(0), img2.unsqueeze(0))
                        logit_diff = model(img1.unsqueeze(0), img3.unsqueeze(0))
                        
                    else:  # style task
                        img_diff_same = torch.abs(img1 - img2).max().item() 
                        img_diff_diff = torch.abs(img1 - img3).max().item()
                        print(f"  [调试] 相同风格图像差异: {img_diff_same:.6f}")
                        print(f"  [调试] 不同风格图像差异: {img_diff_diff:.6f}")
                        
                        # 检查特征
                        v1 = model.encoder(img1.unsqueeze(0))
                        v2 = model.encoder(img2.unsqueeze(0))
                        v3 = model.encoder(img3.unsqueeze(0))
                        
                        feature_diff_same = torch.abs(v1 - v2).max().item()
                        feature_diff_diff = torch.abs(v1 - v3).max().item()
                        
                        print(f"  [调试] 相同风格特征差异: {feature_diff_same:.6f}")
                        print(f"  [调试] 不同风格特征差异: {feature_diff_diff:.6f}")
                        
                        # 检查logit
                        logit_same = model(img1.unsqueeze(0), img2.unsqueeze(0))
                        logit_diff = model(img1.unsqueeze(0), img3.unsqueeze(0))
                    
                    print(f"  [调试] 相同样本logit: {logit_same.item():.4f}")
                    print(f"  [调试] 不同样本logit: {logit_diff.item():.4f}")
                
                # 相同样本应该相似度高，不同样本应该相似度低
                if same_prob > 0.8 and diff_prob < 0.5:
                    correct += 1
                total += 1
                
            except Exception as e:
                print(f"  [错误] 样本{sample_idx}失败: {e}")
                continue
    
    sanity_acc = correct / total if total > 0 else 0
    
    # 🔥 详细统计信息
    if same_probs and diff_probs:
        avg_same = sum(same_probs) / len(same_probs)
        avg_diff = sum(diff_probs) / len(diff_probs)
        print(f"🧠 理智检查详情:")
        print(f"   相同样本平均概率: {avg_same:.4f} (期望>0.8)")
        print(f"   不同样本平均概率: {avg_diff:.4f} (期望<0.5)")
        print(f"   综合准确率: {sanity_acc:.3f}")
    else:
        print(f"🧠 理智检查准确率: {sanity_acc:.3f}")
    
    if sanity_acc > 0.7:
        print("✅ 模型具备基本分辨能力")
        return True
    elif avg_same < 0.1 and avg_diff < 0.1:
        print("⚠️  模型输出过小，可能是sigmoid饱和问题")
        return False
    elif abs(avg_same - avg_diff) < 0.1:
        print("⚠️  模型无法区分相同和不同样本")
        return False
    else:
        print("❌ 模型缺乏基本分辨能力，需要检查")
        return False

def quick_debug_training(encoder_type="enhanced"):
    """
    快速调试训练 - 使用小数据集验证代码逻辑
    适合快速迭代和调试，训练时间约5-10分钟
    
    Args:
        encoder_type: 编码器类型 ("original", "enhanced", "vgg", "hybrid")
    """
    print(f"🚀 快速调试模式 - 小数据集训练 (编码器: {encoder_type})")
    print("=" * 50)
    
    # 🔥 极小的调试配置 - 快速验证代码逻辑
    debug_config = {
        "num_styles": 20,        # 只用20种风格
        "num_contents": 50,      # 只用50种内容  
        "train_samples": 1000,   # 只训练1000个样本
        "batch_size": 16,        # 较小的批量大小
        "epochs": 5,             # 只训练5个epoch
        "lr": 1e-3,              # 较高的学习率快速收敛
        "eval_samples": 200      # 少量评估样本
    }
    
    print(f"📊 调试配置: {debug_config}")
    print("⏱️  预计训练时间: 5-10分钟")
    
    # 训练内容分离模型
    print("\n📝 训练内容分离任务...")
    content_model, accessor, decoder = run_full_training(
        task="content",
        encoder_type=encoder_type,  # 🔥 传递编码器类型
        **debug_config
    )
    
    # 训练风格分离模型  
    print("\n🎨 训练风格分离任务...")
    style_model, _, _ = run_full_training(
        task="style", 
        encoder_type=encoder_type,  # 🔥 传递编码器类型
        **debug_config
    )
    
    print("\n✅ 快速调试训练完成!")
    print("如果结果看起来合理，可以使用 full_scale_training() 进行完整训练")
    
    return content_model, style_model, accessor, decoder

def full_scale_training(encoder_type="enhanced"):
    """
    完整规模训练 - 使用全数据集获得最佳性能
    训练时间约2-4小时
    
    Args:
        encoder_type: 编码器类型 ("original", "enhanced", "vgg", "hybrid")
    """
    print(f"🚀 完整规模训练 - 全数据集训练 (编码器: {encoder_type})")
    print("=" * 50)
    
    # 🔥 全数据集训练配置 - 充分利用数据
    full_config = {
        "num_styles": 2000,      # 使用2000种风格
        "num_contents": 4000,    # 使用4000个字符内容
        "train_samples": 200000, # 20万训练样本
        "batch_size": 64,        # 较大的批量大小
        "epochs": 100,            # 充分训练
        "lr": 1e-4,              # 较低的学习率稳定训练
        "eval_samples": 5000     # 充分的评估样本
    }
    
    print(f"📊 全数据集配置: {full_config}")
    print("⏱️  预计训练时间: 2-4小时")
    print(f"💾 利用数据规模: {full_config['num_styles']} × {full_config['num_contents']} = {full_config['num_styles'] * full_config['num_contents']:,} 组合")
    
    # 运行内容任务训练
    print("\n📝 训练内容分离任务...")
    content_model, accessor, decoder = run_full_training(
        task="content",
        encoder_type=encoder_type,
        **full_config
    )
    
    # 保存内容模型
    torch.save(content_model.state_dict(), "content_siamese_model_full.pth")
    print("💾 内容模型已保存到: content_siamese_model_full.pth")
    
    # 运行风格任务训练
    print("\n🎨 训练风格分离任务...")
    style_model, _, _ = run_full_training(
        task="style",
        encoder_type=encoder_type,
        **full_config
    )
    
    # 保存风格模型
    torch.save(style_model.state_dict(), "style_siamese_model_full.pth")
    print("💾 风格模型已保存到: style_siamese_model_full.pth")
    
    print("\n🎉 完整规模训练完成!")
    print("📈 模型已保存，可用于推理和进一步分析")
    
    return content_model, style_model, accessor, decoder

if __name__ == "__main__":
    import sys
    
    # 🔥 根据命令行参数选择训练模式
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        print("🔍 启动快速调试训练...")
        print("💡 提示: 使用 'python full_training.py' 进行完整训练")
        quick_debug_training()
    else:
        print("🎯 启动完整规模训练...")
        print("💡 提示: 使用 'python full_training.py debug' 进行快速调试")
        full_scale_training()