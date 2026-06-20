import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from datasets.ucsd_dataset import UCSDPed2Dataset

from models.conv_memae_extended import (
    ConvMemAEExtended
)


MEMORY_SIZE = 500

EPOCHS = 50


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

dataset = UCSDPed2Dataset(
    "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train"
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

model = ConvMemAEExtended(
    memory_size=MEMORY_SIZE
).to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0

    for images in loader:

        images = images.to(
            device
        )

        optimizer.zero_grad()

        (
            recon,
            att,
            z,
            z_mem
        ) = model(images)

        recon_loss = criterion(
            recon,
            images
        )

        entropy_loss = -(
            att *
            torch.log(
                att + 1e-12
            )
        ).sum(
            dim=1
        ).mean()

        compactness_loss = (
            (z - z_mem) ** 2
        ).mean()

        loss = (
            recon_loss
            +
            0.0002 * entropy_loss
            +
            0.01 * compactness_loss
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS}"
        f" Loss={epoch_loss/len(loader):.6f}"
    )

torch.save(
    model.state_dict(),
    f"conv_memae_{MEMORY_SIZE}.pth"
)

print("Training finished.")