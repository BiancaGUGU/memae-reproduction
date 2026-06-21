import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from models.memae import MemAE


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True
)

model = MemAE().to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3
)

epochs = 5

for epoch in range(epochs):

    running_loss = 0.0
    running_recon = 0.0
    running_entropy = 0.0   
    for images, _ in loader:

        images = images.view(
            images.size(0),
            -1
        ).to(device)

        optimizer.zero_grad()

        # recon, att = model(images)

        # loss = criterion(
        #     recon,
        #     images
        # )
        recon, att = model(images)

        recon_loss = criterion(
            recon,
            images
        )

        entropy_loss = -(
            att * torch.log(att + 1e-12)
        ).sum(dim=1).mean()

        loss = (
            recon_loss
            + 0.0002 * entropy_loss
        )
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        running_recon += recon_loss.item()
        running_entropy += entropy_loss.item()
    # avg_loss = running_loss / len(loader)

    # print(
    #     f"Epoch [{epoch+1}/{epochs}] "
    #     f"Loss: {avg_loss:.6f}"
    # )
    avg_loss = running_loss / len(loader)
    avg_recon = running_recon / len(loader)
    avg_entropy = running_entropy / len(loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss={avg_loss:.6f} "
        f"Recon={avg_recon:.6f} "
        f"Entropy={avg_entropy:.6f}"
    )
# print("Training finished!")

torch.save(
    model.state_dict(),
    "memae_model.pth"
)

print("Model saved!")