import logging
from typing import Optional
from urllib.error import URLError
from ssl import SSLError

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights, VGG19_Weights


class VGGEncoder(nn.Module):
    """VGG-based encoder using pretrained weights"""
    def __init__(
        self,
        in_ch=1,
        emb_dim=512,
        vgg_variant="vgg16",
        task="content",
        weights=None,
    ):
        super().__init__()
        self.task = task

        if vgg_variant not in ("vgg16", "vgg19"):
            raise ValueError(f"Unsupported VGG variant: {vgg_variant}")

        weight_enum = weights
        if weights is None:
            if vgg_variant == "vgg16":
                weight_enum = VGG16_Weights.DEFAULT
            else:
                weight_enum = VGG19_Weights.DEFAULT

        try:
            if vgg_variant == "vgg16":
                vgg = models.vgg16(weights=weight_enum)
            else:
                vgg = models.vgg19(weights=weight_enum)
        except (URLError, SSLError, RuntimeError) as exc:
            logging.warning(
                "Falling back to randomly initialised %s because pretrained weights could not be loaded: %s",
                vgg_variant,
                exc,
            )
            if vgg_variant == "vgg16":
                vgg = models.vgg16(weights=None)
            else:
                vgg = models.vgg19(weights=None)

        self.features = vgg.features
        for module in self.features.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        if in_ch == 1:
            first_conv = self.features[0]
            self.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
            with torch.no_grad():
                mean_weight = first_conv.weight.mean(dim=1, keepdim=True)
                self.features[0].weight.copy_(mean_weight)
                if first_conv.bias is not None:
                    self.features[0].bias.copy_(first_conv.bias)

        self._freeze_early_layers(freeze_layers=3)

        if task == "content":
            self.feature_layers = [10, 17, 24]  # conv2_2, conv3_4, conv4_4
        else:  # style
            self.feature_layers = [3, 8, 15, 22, 29]  # conv1_2, conv2_2, conv3_3, conv4_3, conv5_3

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4)) if task == "content" else nn.AdaptiveAvgPool2d((1, 1))

        self.emb_dim = emb_dim
        self.fc: Optional[nn.Sequential] = None

    def _freeze_early_layers(self, freeze_layers=3):
        """Freeze early layers"""
        for i, module in enumerate(self.features.children()):
            if i < freeze_layers:
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.size(1) not in (1, 3):
            x = x.mean(dim=1, keepdim=True)

        if x.size(1) == 1 and self.features[0].in_channels == 3:
            x = x.repeat(1, 3, 1, 1)

        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_layers:
                pooled = self.adaptive_pool(x)
                features.append(pooled.view(pooled.size(0), -1))

        if features:
            combined_features = torch.cat(features, dim=1)
        else:
            combined_features = self.adaptive_pool(x).view(x.size(0), -1)

        if self.fc is None:
            self._build_fc(combined_features.size(1), combined_features.device)

        return self.fc(combined_features)

    def _build_fc(self, feature_dim: int, device: torch.device) -> None:
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, self.emb_dim * 2),
            nn.BatchNorm1d(self.emb_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.emb_dim * 2, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        ).to(device)



class EnhancedContentEncoder(nn.Module):
    """Enhanced content encoder"""
    def __init__(self, in_ch=1, emb_dim=512):
        super().__init__()

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

            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.spatial_pool = nn.AdaptiveAvgPool2d(4)

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
    """Enhanced style encoder"""
    def __init__(self, in_ch=1, emb_dim=512):
        super().__init__()

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

            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        self.global_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
            nn.AdaptiveAvgPool2d(2),
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

        pooled_features = []
        for pool in self.global_pools:
            pooled = pool(h).view(h.size(0), -1)
            pooled_features.append(pooled)

        h = torch.cat(pooled_features, dim=1)
        return self.classifier(h)



class SiameseJudge(nn.Module):
    def __init__(self, in_ch=1, emb_dim=512, mlp_hidden=512, task="content", encoder_type="enhanced"):
        super().__init__()

        if encoder_type == "enhanced":
            if task == "content":
                self.encoder = EnhancedContentEncoder(in_ch=in_ch, emb_dim=emb_dim)
            else:
                self.encoder = EnhancedStyleEncoder(in_ch=in_ch, emb_dim=emb_dim)
        elif encoder_type == "vgg":
            self.encoder = VGGEncoder(in_ch=in_ch, emb_dim=emb_dim, task=task)
        else:
            raise ValueError(f"encoder_type must be 'enhanced' or 'vgg', got {encoder_type}")

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
        return logit.squeeze()
