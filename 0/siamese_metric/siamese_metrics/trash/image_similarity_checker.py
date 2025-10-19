#!/usr/bin/env python3
"""
图片相似度检查器 - 使用训练好的Siamese网络判断两张图片是否相似
作者: gz2199
用途: 直接输入两张图片，通过Siamese网络判断相似度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import argparse
import os
from typing import Tuple, Optional, Union

# 复制模型定义（与训练时保持一致）
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
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            
            # Block 4
            nn.Conv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(512, emb_dim * 2),
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
        h = self.global_pool(h)
        h = h.view(h.size(0), -1)
        return self.classifier(h)

class SiameseJudge(nn.Module):
    """Siamese网络 - 与训练时完全一致"""
    def __init__(self, in_ch=1, emb_dim=512, mlp_hidden=512, task="content", encoder_type="enhanced"):
        super().__init__()
        
        # 🔥 支持多种编码器类型
        if encoder_type == "enhanced":
            # 增强版编码器 - VGG风格的深层网络
            if task == "content":
                self.encoder = EnhancedContentEncoder(in_ch=in_ch, emb_dim=emb_dim)
            else:  # style
                self.encoder = EnhancedStyleEncoder(in_ch=in_ch, emb_dim=emb_dim)
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
        return torch.sigmoid(logit).squeeze()  # 直接返回概率

class ImageSimilarityChecker:
    """图片相似度检查器"""
    
    def __init__(self, 
                 content_model_path: str = "10kcontent_siamese_model_full.pth",
                 style_model_path: str = "10kstyle_siamese_model_full.pth",
                 device: str = "auto"):
        """
        初始化相似度检查器
        
        Args:
            content_model_path: 内容模型路径
            style_model_path: 风格模型路径
            device: 设备 ("auto", "cuda", "cpu")
        """
        self.device = self._get_device(device)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 调整到训练时的尺寸
            transforms.Grayscale(num_output_channels=1),  # 转为灰度图
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1, 1]
        ])
        
        # 加载模型
        self.content_model = self._load_model(content_model_path, "content")
        self.style_model = self._load_model(style_model_path, "style")
        
        print(f"✅ 相似度检查器初始化完成 (设备: {self.device})")
    
    def _get_device(self, device: str) -> torch.device:
        """获取计算设备"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_model(self, model_path: str, task: str) -> SiameseJudge:
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            # 尝试调试模型路径
            debug_path = f"debug_{task}_siamese_model.pth"
            if os.path.exists(debug_path):
                model_path = debug_path
                print(f"⚠️  使用调试模型: {debug_path}")
            else:
                raise FileNotFoundError(f"找不到模型文件: {model_path}")
        
        # 创建模型 - 使用与训练时相同的架构
        model = SiameseJudge(
            in_ch=1, 
            emb_dim=512, 
            mlp_hidden=512, 
            task=task, 
            encoder_type="enhanced"
        )
        
        # 加载权重
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"✅ {task}模型加载成功: {model_path}")
        except Exception as e:
            print(f"❌ {task}模型加载失败: {e}")
            raise
        
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """预处理图像"""
        # 处理不同类型的输入
        if isinstance(image_input, str):
            # 文件路径
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"图片文件不存在: {image_input}")
            image = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            # numpy数组
            image = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            # PIL图像
            image = image_input
        else:
            raise ValueError(f"不支持的图像类型: {type(image_input)}")
        
        # 应用预处理
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)  # 添加batch维度
    
    def check_content_similarity(self, 
                                image1: Union[str, Image.Image, np.ndarray],
                                image2: Union[str, Image.Image, np.ndarray],
                                threshold: float = 0.5) -> Tuple[bool, float]:
        """
        检查两张图片的内容相似度
        
        Args:
            image1: 第一张图片
            image2: 第二张图片
            threshold: 相似度阈值
            
        Returns:
            (是否相似, 相似度分数)
        """
        with torch.no_grad():
            img1_tensor = self.preprocess_image(image1)
            img2_tensor = self.preprocess_image(image2)
            
            similarity_score = self.content_model(img1_tensor, img2_tensor).item()
            is_similar = similarity_score > threshold
            
            return is_similar, similarity_score
    
    def check_style_similarity(self, 
                              image1: Union[str, Image.Image, np.ndarray],
                              image2: Union[str, Image.Image, np.ndarray],
                              threshold: float = 0.5) -> Tuple[bool, float]:
        """
        检查两张图片的风格相似度
        
        Args:
            image1: 第一张图片
            image2: 第二张图片
            threshold: 相似度阈值
            
        Returns:
            (是否相似, 相似度分数)
        """
        with torch.no_grad():
            img1_tensor = self.preprocess_image(image1)
            img2_tensor = self.preprocess_image(image2)
            
            similarity_score = self.style_model(img1_tensor, img2_tensor).item()
            is_similar = similarity_score > threshold
            
            return is_similar, similarity_score
    
    def comprehensive_check(self, 
                           image1: Union[str, Image.Image, np.ndarray],
                           image2: Union[str, Image.Image, np.ndarray],
                           content_threshold: float = 0.5,
                           style_threshold: float = 0.5) -> dict:
        """
        全面检查两张图片的相似度（内容+风格）
        
        Args:
            image1: 第一张图片
            image2: 第二张图片
            content_threshold: 内容相似度阈值
            style_threshold: 风格相似度阈值
            
        Returns:
            包含详细结果的字典
        """
        # 检查内容相似度
        content_similar, content_score = self.check_content_similarity(
            image1, image2, content_threshold
        )
        
        # 检查风格相似度
        style_similar, style_score = self.check_style_similarity(
            image1, image2, style_threshold
        )
        
        # 综合判断
        overall_similar = content_similar and style_similar
        overall_score = (content_score + style_score) / 2
        
        return {
            "content": {
                "similar": content_similar,
                "score": content_score,
                "threshold": content_threshold
            },
            "style": {
                "similar": style_similar,
                "score": style_score,
                "threshold": style_threshold
            },
            "overall": {
                "similar": overall_similar,
                "score": overall_score
            }
        }

def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="图片相似度检查器")
    parser.add_argument("image1", help="第一张图片路径")
    parser.add_argument("image2", help="第二张图片路径")
    parser.add_argument("--mode", choices=["content", "style", "both"], 
                       default="both", help="检查模式")
    parser.add_argument("--content-threshold", type=float, default=0.5,
                       help="内容相似度阈值")
    parser.add_argument("--style-threshold", type=float, default=0.5,
                       help="风格相似度阈值")
    parser.add_argument("--content-model", 
                       default="10kcontent_siamese_model_full.pth",
                       help="内容模型路径")
    parser.add_argument("--style-model", 
                       default="10kstyle_siamese_model_full.pth",
                       help="风格模型路径")
    
    args = parser.parse_args()
    
    try:
        # 初始化检查器
        checker = ImageSimilarityChecker(
            content_model_path=args.content_model,
            style_model_path=args.style_model
        )
        
        print(f"\n🔍 比较图片:")
        print(f"  图片1: {args.image1}")
        print(f"  图片2: {args.image2}")
        print(f"  模式: {args.mode}")
        print("=" * 60)
        
        if args.mode == "content":
            # 只检查内容相似度
            is_similar, score = checker.check_content_similarity(
                args.image1, args.image2, args.content_threshold
            )
            print(f"📝 内容相似度: {score:.4f}")
            print(f"   阈值: {args.content_threshold}")
            print(f"   结果: {'✅ 相似' if is_similar else '❌ 不相似'}")
            
        elif args.mode == "style":
            # 只检查风格相似度
            is_similar, score = checker.check_style_similarity(
                args.image1, args.image2, args.style_threshold
            )
            print(f"🎨 风格相似度: {score:.4f}")
            print(f"   阈值: {args.style_threshold}")
            print(f"   结果: {'✅ 相似' if is_similar else '❌ 不相似'}")
            
        else:  # both
            # 全面检查
            result = checker.comprehensive_check(
                args.image1, args.image2, 
                args.content_threshold, args.style_threshold
            )
            
            print(f"📝 内容相似度: {result['content']['score']:.4f}")
            print(f"   阈值: {result['content']['threshold']}")
            print(f"   结果: {'✅ 相似' if result['content']['similar'] else '❌ 不相似'}")
            print()
            print(f"🎨 风格相似度: {result['style']['score']:.4f}")
            print(f"   阈值: {result['style']['threshold']}")
            print(f"   结果: {'✅ 相似' if result['style']['similar'] else '❌ 不相似'}")
            print()
            print(f"🎯 综合相似度: {result['overall']['score']:.4f}")
            print(f"   综合结果: {'✅ 相似' if result['overall']['similar'] else '❌ 不相似'}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())