import torch

from models.memae import MemAE

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = MemAE().to(device)

model.load_state_dict(
    torch.load(
        "memae_model.pth",
        map_location=device
    )
)

model.eval()

x = torch.randn(1, 784).to(device)

with torch.no_grad():

    recon, att = model(x)

print("Input:", x.shape)
print("Output:", recon.shape)