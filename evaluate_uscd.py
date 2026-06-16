import time

import torch
import torch.nn as nn

from pathlib import Path

from PIL import Image

from torchvision import transforms

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

from models.conv_memae import ConvMemAE


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

model = ConvMemAE().to(device)

model.load_state_dict(
    torch.load(
        "conv_memae_ucsd.pth",
        map_location=device
    )
)

model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

criterion = nn.MSELoss(
    reduction="mean"
)

test_root = Path(
    "/home/bianca/memae-reproduction/data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test"
)

all_scores = []
all_labels = []

num_frames = 0

start_time = time.time()

for test_dir in sorted(test_root.glob("Test0*")):

    if test_dir.name.endswith("_gt"):
        continue

    gt_dir = Path(
        str(test_dir) + "_gt"
    )

    print(
        f"Processing {test_dir.name}"
    )

    video_scores = []
    video_labels = []

    frame_paths = sorted(
        test_dir.glob("*.tif")
    )

    gt_paths = sorted(
        gt_dir.glob("*.bmp")
    )

    for frame_path, gt_path in zip(
        frame_paths,
        gt_paths
    ):

        image = Image.open(
            frame_path
        )

        image = transform(
            image
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            recon, _ = model(image)

            score = criterion(
                recon,
                image
            ).item()

        gt = Image.open(
            gt_path
        ).convert("L")

        gt_pixels = list(
            gt.getdata()
        )

        label = (
            1 if max(gt_pixels) > 0
            else 0
        )

        video_scores.append(score)
        video_labels.append(label)

        all_scores.append(score)
        all_labels.append(label)

        num_frames += 1

    video_auc = roc_auc_score(
        video_labels,
        video_scores
    )

    print(
        f"{test_dir.name} AUC: "
        f"{video_auc:.4f}"
    )

    plt.figure(figsize=(10, 4))

    plt.plot(video_scores)

    plt.title(
        f"{test_dir.name} Scores"
    )

    plt.xlabel("Frame")

    plt.ylabel(
        "Reconstruction Error"
    )

    plt.tight_layout()

    plt.savefig(
        f"{test_dir.name}_scores.png"
    )

    plt.close()

elapsed = (
    time.time() - start_time
)

fps = (
    num_frames / elapsed
)

auc = roc_auc_score(
    all_labels,
    all_scores
)

print()
print(
    f"FRAME-LEVEL AUC: {auc:.6f}"
)

print(
    f"FPS: {fps:.2f}"
)