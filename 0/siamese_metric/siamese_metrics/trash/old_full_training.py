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
                    print(f"  - 调整为: {self.num_fonts}字体 × {self.num_characters}字符")
                elif self.total_samples % 2056 == 0:
                    self.num_characters = self.total_samples // 2056
                    print(f"  - 调整为: {self.num_fonts}字体 × {self.num_characters}字符")
                else:
                    # 如果都不匹配，使用平方根近似
                    import math
                    side = int(math.sqrt(self.total_samples))
                    self.num_fonts = side
                    self.num_characters = self.total_samples // side
                    print(f"  - 近似调整为: {self.num_fonts}字体 × {self.num_characters}字符")
                    
            self.num_styles_per_char = self.num_fonts
            self.fallback_mode = True
            
            print(f"[ImprovedLatentAccessor] Fallback模式配置:")
            print(f"  - 字体数量: {self.num_fonts}")
            print(f"  - 字符数量: {self.num_characters}")
            print(f"  - 数据组织: font_idx * {self.num_characters} + char_idx")
            
    def get_by_indices(self, font_idx: int, char_idx: int) -> torch.Tensor:
        """根据字体和字符索引获取潜在向量"""
        if hasattr(self, 'dataset') and not self.fallback_mode:
            # 使用FourWayDataset方式
            return self.dataset.get_latent_by_indices(font_idx, char_idx).to(self.device)
        else:
            # 使用fallback方式
            linear_idx = font_idx * self.num_characters + char_idx
            
            if hasattr(self, 'latents_hwc'):
                # 4D格式 (N, H, W, C)
                latent_hwc = self.latents_hwc[linear_idx]  # (H, W, C)
                latent_chw = latent_hwc.permute(2, 0, 1)   # (C, H, W)
                return latent_chw.to(self.device)
            elif hasattr(self, 'raw_tensor'):
                return self.raw_tensor[linear_idx].to(self.device)
            else:
                raise RuntimeError("No latent data available")
    
    def get_by_names(self, font_name: str, char: str) -> torch.Tensor:
        """根据字体名称和字符获取潜在向量（仅FourWay模式支持）"""
        if hasattr(self, 'dataset') and not self.fallback_mode:
            font_idx = self.font_to_idx.get(font_name)
            char_idx = self.char_to_idx.get(char)
            
            if font_idx is None:
                raise ValueError(f"Unknown font: {font_name}")
            if char_idx is None:
                raise ValueError(f"Unknown character: {char}")
                
            return self.get_by_indices(font_idx, char_idx)
        else:
            raise NotImplementedError("get_by_names only available in FourWay mode")
    
    def random_sample(self) -> Tuple[torch.Tensor, int, int]:
        """随机采样一个潜在向量"""
        font_idx = random.randint(0, self.num_fonts - 1)
        char_idx = random.randint(0, self.num_characters - 1)
        return self.get_by_indices(font_idx, char_idx), font_idx, char_idx
        
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

class TinyEncoder(nn.Module):
    def __init__(self, in_ch=1, emb_dim=512):
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
    def __init__(self, in_ch=1, emb_dim=512, mlp_hidden=512):
        super().__init__()
        self.encoder = TinyEncoder(in_ch=in_ch, emb_dim=emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, mlp_hidden), 
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, mlp_hidden//2),
            nn.BatchNorm1d(mlp_hidden//2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden//2, 1)
        )

    def forward(self, x1, x2):
        v1 = self.encoder(x1)
        v2 = self.encoder(x2)
        diff = torch.abs(v1 - v2)
        logit = self.head(diff)
        return logit, v1, v2

@torch.no_grad()
def score_pair(model: nn.Module, img1: torch.Tensor, img2: torch.Tensor, device=None) -> float:
    device = device or next(model.parameters()).device
    model.eval()
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)
    logit, _, _ = model(img1, img2)
    prob = torch.sigmoid(logit).item()
    return prob

class FullDatasetPairDataset(Dataset):
    """
    使用完整数据集的配对数据集
    从大量样本中随机选择配对进行训练
    """
    def __init__(self, 
                 accessor: ImprovedLatentAccessor, 
                 decoder: nn.Module,
                 cfg: dict,
                 device: torch.device,
                 task: Literal["content","style"]="content",
                 num_styles: int = 100,
                 num_contents: int = 1000,
                 length: int = 50000,  # 增加训练样本数
                 augment: bool = True):
        """
        Args:
            accessor: 改进的潜在编码访问器
            decoder: VAE解码器模型
            cfg: VAE配置字典
            device: 计算设备
            task: 任务类型 ('content' 或 'style')
            num_styles: 使用的风格数量
            num_contents: 使用的内容数量  
            length: 训练样本总数
            augment: 是否使用数据增强
        """
        self.accessor = accessor
        self.decoder = decoder
        self.cfg = cfg
        self.device = device
        self.task = task
        self.num_styles = num_styles
        self.num_contents = num_contents
        self.length = length
        self.augment = augment
        
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
        """获取训练样本对"""
        # 随机选择索引
        content_i = random.randint(0, self.num_contents - 1)
        content_j = random.randint(0, self.num_contents - 1)
        style_p = random.randint(0, self.num_styles - 1)
        style_q = random.randint(0, self.num_styles - 1)
        
        # 确保有不同的选择用于负样本
        while content_j == content_i:
            content_j = random.randint(0, self.num_contents - 1)
        while style_q == style_p:
            style_q = random.randint(0, self.num_styles - 1)
        
        try:
            # 获取潜在编码
            z1 = self.accessor.get(content_i, style_p)
            if idx < 5:  # 只在前几个样本打印调试信息
                print(f"[DEBUG] Sample {idx}: z1.shape = {z1.shape}")
            
            # 解码为图像 (保持在CPU上)
            ci_sp = decode_to_image(self.decoder, z1, self.cfg, self.device)
            if idx < 5:
                print(f"[DEBUG] Sample {idx}: ci_sp.shape = {ci_sp.shape}")
            
            # 50/50 正负样本
            is_positive = (random.random() < 0.5)
            
            if self.task == "content":
                if is_positive:
                    # 正样本: 相同内容，不同风格
                    z2 = self.accessor.get(content_i, style_q)
                    ci_sq = decode_to_image(self.decoder, z2, self.cfg, self.device)
                    x1, x2, y = ci_sp, ci_sq, 1.0
                else:
                    # 负样本: 不同内容，相同风格
                    z2 = self.accessor.get(content_j, style_p)
                    cj_sp = decode_to_image(self.decoder, z2, self.cfg, self.device)
                    x1, x2, y = ci_sp, cj_sp, 0.0
            else:  # style task
                if is_positive:
                    # 正样本: 相同风格，不同内容
                    z2 = self.accessor.get(content_j, style_p)
                    cj_sp = decode_to_image(self.decoder, z2, self.cfg, self.device)
                    x1, x2, y = ci_sp, cj_sp, 1.0
                else:
                    # 负样本: 不同风格，相同内容
                    z2 = self.accessor.get(content_i, style_q)
                    ci_sq = decode_to_image(self.decoder, z2, self.cfg, self.device)
                    x1, x2, y = ci_sp, ci_sq, 0.0

            # 应用数据增强
            x1 = self._aug(x1)
            x2 = self._aug(x2)
            
            # 确保张量是连续的并且可以被复制
            x1 = x1.contiguous().detach().clone()
            x2 = x2.contiguous().detach().clone()
            
            return x1, x2, torch.tensor([y], dtype=torch.float32)
            
        except Exception as e:
            print(f"[WARN] 样本生成失败 (idx={idx}): {e}")
            # 返回一个dummy样本避免训练中断
            dummy_img = torch.zeros(1, 32, 32)
            return dummy_img, dummy_img, torch.tensor([0.0], dtype=torch.float32)

def run_full_training(
    task: Literal["content","style"]="content",
    local_path: Optional[str] = None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    
    # 训练参数
    num_styles: int = 500,      # 使用的风格数量
    num_contents: int = 2000,   # 使用的内容数量
    train_samples: int = 100000, # 训练样本数
    batch_size: int = 64,       # 批量大小
    epochs: int = 20,           # 训练轮数
    lr: float = 1e-4,           # 学习率
    
    # 评估参数
    eval_samples: int = 1000,   # 评估样本数
):
    """
    运行完整数据集训练
    """
    print("🚀 开始全数据集字体训练...")
    print(f"📊 训练配置:")
    print(f"  - 任务: {task}")
    print(f"  - 设备: {device}")
    print(f"  - 风格数: {num_styles}")
    print(f"  - 内容数: {num_contents}")
    print(f"  - 训练样本: {train_samples}")
    print(f"  - 批量大小: {batch_size}")
    print(f"  - 训练轮数: {epochs}")
    
    # 1. 加载数据
    print("\n📥 加载潜在编码数据...")
    # 使用ImprovedLatentAccessor替代原来的LatentAccessor
    accessor = ImprovedLatentAccessor(
        pt_path=local_path,
        chars_path="/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0/char_list.txt",
        fonts_json="/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0/lmdb_keys.json",
        device="cpu"
    )
    
    # 2. 加载VAE解码器
    print("\n🔧 加载VAE解码器...")
    decoder, cfg, device_used = load_vae_decoder(device=device)
    
    # 3. 创建数据集和加载器
    print("\n📚 创建训练数据集...")
    train_dataset = FullDatasetPairDataset(
        accessor=accessor,
        decoder=decoder,
        cfg=cfg,
        device=device_used,
        task=task,
        num_styles=num_styles,
        num_contents=num_contents,
        length=train_samples,
        augment=True
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
    
    model = SiameseJudge(in_ch=C, emb_dim=512, mlp_hidden=512)  # 增大模型容量
    
    # 5. 训练模型
    print("\n🎯 开始训练...")
    model = train_full_model(model, train_loader, device=device, lr=lr, epochs=epochs)
    
    # 6. 评估模型
    print("\n📊 评估模型性能...")
    eval_model(model, accessor, decoder, cfg, device_used, task, num_styles, num_contents, eval_samples)
    
    return model, accessor, decoder

def train_full_model(model, loader, device="cuda", lr=1e-4, epochs=20):
    """训练完整模型"""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.BCEWithLogitsLoss()
    
    for ep in range(1, epochs+1):
        model.train()
        tot_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (x1, x2, y) in enumerate(loader):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            
            logit, _, _ = model(x1, x2)
            loss = crit(logit, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # 计算准确率
            pred = (torch.sigmoid(logit) > 0.5).float()
            correct += (pred == y).sum().item()
            total += y.size(0)
            tot_loss += loss.item() * x1.size(0)
            
            # 每100个batch打印一次进度
            if (batch_idx + 1) % 100 == 0:
                avg_loss = tot_loss / (batch_idx + 1) / loader.batch_size
                acc = correct / total
                print(f"  Batch {batch_idx+1}/{len(loader)}: loss={avg_loss:.4f}, acc={acc:.3f}")
        
        scheduler.step()
        
        avg_loss = tot_loss / len(loader.dataset)
        acc = correct / total
        lr_current = scheduler.get_last_lr()[0]
        print(f"[Epoch {ep}/{epochs}] loss={avg_loss:.4f}, acc={acc:.3f}, lr={lr_current:.2e}")
    
    return model

def eval_model(model, accessor, decoder, cfg, device, task, num_styles, num_contents, eval_samples):
    """评估模型性能"""
    model.eval()
    correct = 0
    total = 0
    
    print(f"🔍 评估 {eval_samples} 个样本...")
    
    with torch.no_grad():
        for _ in range(eval_samples):
            # 随机选择测试样本
            i = random.randint(0, num_contents - 1)
            j = random.randint(0, num_contents - 1)
            p = random.randint(0, num_styles - 1)
            q = random.randint(0, num_styles - 1)
            
            while j == i:
                j = random.randint(0, num_contents - 1)
            while q == p:
                q = random.randint(0, num_styles - 1)
            
            try:
                # 生成测试图像
                ci_sp = decode_to_image(decoder, accessor.get(i, p), cfg, device).to(device)
                ci_sq = decode_to_image(decoder, accessor.get(i, q), cfg, device).to(device)
                cj_sp = decode_to_image(decoder, accessor.get(j, p), cfg, device).to(device)
                
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
                    correct += 1
                if neg_prob < 0.5:
                    correct += 1
                total += 2
                
            except Exception as e:
                continue
    
    accuracy = correct / total if total > 0 else 0
    print(f"✅ 评估完成: 准确率 = {accuracy:.3f} ({correct}/{total})")
    
    return accuracy

if __name__ == "__main__":
    # 配置参数
    local_path = "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/Font-Latent-Full-PT/font_latents_v2.pt"
    
    # 运行内容任务训练
    print("=" * 60)
    print("🔤 训练内容分离任务")
    print("=" * 60)
    
    content_model, accessor, decoder = run_full_training(
        task="content",
        local_path=local_path,
        num_styles=200,      # 使用200种风格
        num_contents=500,    # 使用500种内容
        train_samples=50000, # 5万训练样本
        batch_size=32,       # 批量大小32
        epochs=15,           # 15个epoch
        lr=1e-4,
        eval_samples=1000
    )
    
    # 保存内容模型
    torch.save(content_model.state_dict(), "old_content_siamese_model.pth")
    print("💾 内容模型已保存到: old_content_siamese_model.pth")
    
    print("\n" + "=" * 60)
    print("🎨 训练风格分离任务")
    print("=" * 60)
    
    # 运行风格任务训练
    style_model, _, _ = run_full_training(
        task="style",
        local_path=local_path,
        num_styles=200,
        num_contents=500,
        train_samples=50000,
        batch_size=32,
        epochs=15,
        lr=5e-5,
        eval_samples=10000
    )
    
    # 保存风格模型
    torch.save(style_model.state_dict(), "old_style_siamese_model.pth")
    print("💾 风格模型已保存到: old_style_siamese_model.pth")
    
    print("\n🎉 全数据集训练完成!")