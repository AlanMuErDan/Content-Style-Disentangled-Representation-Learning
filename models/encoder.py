import torch.nn as nn
from torchvision.models import resnet18

class ResNetEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        base = resnet18(weights=None)

        base.conv1 = nn.Conv2d(
            in_channels=1,  # Change input channels to 1 for grayscale images
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.features = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out = nn.Linear(512, output_dim)

    def forward(self, x):
        feat = self.features(x)
        pooled = self.pool(feat).squeeze()
        return self.out(pooled)

def build_encoder(name="resnet18", output_dim=256):
    if name == "resnet18":
        return ResNetEncoder(output_dim=output_dim)
    else:
        raise NotImplementedError(f"Unknown encoder: {name}")