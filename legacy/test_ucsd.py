import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from datasets.ucsd_dataset import UCSDPed2Dataset
from models.conv_memae import ConvMemAE


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

dataset = UCSDPed2Dataset(
    "/home/bianca/memae-reproduction/data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test"
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

model = ConvMemAE().to(device)

model.load_state_dict(
    torch.load(
        "conv_memae_ucsd.pth",
        map_location=device
    )
)

model.eval()

scores = []

with torch.no_grad():

    for image in loader:

        image = image.to(device)

        recon, _ = model(image)

        mse = torch.mean(
            (image - recon) ** 2
        )

        scores.append(
            mse.item()
        )

print()

print("Frames:", len(scores))

print(
    "Min score:",
    min(scores)
)

print(
    "Max score:",
    max(scores)
)

print(
    "Mean score:",
    sum(scores) / len(scores)
)

import matplotlib.pyplot as plt

plt.plot(scores)

plt.xlabel("Frame")
plt.ylabel("Reconstruction Error")

plt.title("UCSD Ped2 Anomaly Scores")

plt.show()