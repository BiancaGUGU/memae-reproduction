import torch

from models.conv_memae import ConvMemAE

model = ConvMemAE()

x = torch.randn(
    4,
    1,
    128,
    128
)

recon, att = model(x)

print(x.shape)
print(recon.shape)
print(att.shape)