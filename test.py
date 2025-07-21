from models.discriminator import SimplePatchDiscriminator
import torch

D = SimplePatchDiscriminator(in_channels=1)
x = torch.randn(4, 1, 64, 64)  # batch size = 4, 1通道, 64x64图像
y = D(x)
print(y.shape)
