import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from datasets.ucsd_dataset import UCSDPed2Dataset
from models.conv_memae import ConvMemAE


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

dataset = UCSDPed2Dataset(
    "/home/bianca/memae-reproduction/data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train"
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

model = ConvMemAE().to(device)

try:

    model.load_state_dict(
        torch.load(
            "conv_memae_ucsd.pth",
            map_location=device
        )
    )

    print("Loaded existing model.")

except FileNotFoundError:

    print("Training from scratch.")

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)

epochs = 20

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images in loader:

        images = images.to(device)

        optimizer.zero_grad()

        recon, att = model(images)

        loss = criterion(
            recon,
            images
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = (
        running_loss /
        len(loader)
    )

    print(
        f"Epoch {epoch+1}: "
        f"{avg_loss:.6f}"
    )

    torch.save(
        model.state_dict(),
        "conv_memae_ucsd.pth"
    )

print("Model saved!")