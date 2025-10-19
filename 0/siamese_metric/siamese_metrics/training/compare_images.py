import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os
import sys

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

class ImageComparator:
    def __init__(self, content_model_path, style_model_path, device="auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else "cpu")
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.content_model = self._load_model(content_model_path, "content")
        self.style_model = self._load_model(style_model_path, "style")
        
    def _load_model(self, model_path, task):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = SiameseJudge(in_ch=1, emb_dim=512, mlp_hidden=512, task=task, encoder_type="enhanced")
        
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 检查checkpoint格式并提取model_state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 这是完整的training checkpoint，提取model_state_dict
            state_dict = checkpoint['model_state_dict']
            print(f"✅ Loaded training checkpoint for {task} task")
            print(f"   - Training timestamp: {checkpoint.get('train_timestamp', 'unknown')}")
            print(f"   - Epochs completed: {checkpoint.get('epochs_completed', 'unknown')}")
            if 'final_train_accuracy' in checkpoint:
                print(f"   - Final training accuracy: {checkpoint['final_train_accuracy']:.4f}")
            if 'final_test_accuracy' in checkpoint:
                print(f"   - Final test accuracy: {checkpoint['final_test_accuracy']:.4f}")
        else:
            # 这是纯模型权重
            state_dict = checkpoint
            print(f"✅ Loaded model weights for {task} task")
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model
    
    def _preprocess_image(self, image_path):
        image = Image.open(image_path)
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)
    
    def compare_images(self, image1_path, image2_path, threshold=0.5):
        with torch.no_grad():
            img1 = self._preprocess_image(image1_path)
            img2 = self._preprocess_image(image2_path)
            
            content_logit = self.content_model(img1, img2)
            style_logit = self.style_model(img1, img2)
            
            content_score = torch.sigmoid(content_logit).item()
            style_score = torch.sigmoid(style_logit).item()
            
            content_similar = content_score > threshold
            style_similar = style_score > threshold
            
            return {
                "content_score": content_score,
                "style_score": style_score,
                "content_similar": content_similar,
                "style_similar": style_similar,
                "overall_score": (content_score + style_score) / 2,
                "overall_similar": content_similar and style_similar
            }

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_images.py <image1_path> <image2_path>")
        return
    
    image1_path = sys.argv[1]
    image2_path = sys.argv[2]
    
    content_model_path = "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0/10kcontent_siamese_model_full.pth"
    style_model_path = "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0/10kstyle_siamese_model_full.pth"
    
    try:
        comparator = ImageComparator(content_model_path, style_model_path)
        result = comparator.compare_images(image1_path, image2_path)
        
        print(f"Image 1: {image1_path}")
        print(f"Image 2: {image2_path}")
        print(f"Content similarity: {result['content_score']:.4f} ({'Similar' if result['content_similar'] else 'Different'})")
        print(f"Style similarity: {result['style_score']:.4f} ({'Similar' if result['style_similar'] else 'Different'})")
        print(f"Overall similarity: {result['overall_score']:.4f} ({'Similar' if result['overall_similar'] else 'Different'})")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()