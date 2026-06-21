import time

import torch
import torch.nn as nn

from pathlib import Path

from PIL import Image

from torchvision import transforms

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

from models.conv_memae_extended import (
    ConvMemAEExtended
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

CHECKPOINT = "best_checkpoint_500.pth"

print(f"Loading checkpoint: {CHECKPOINT}")

model = ConvMemAEExtended(
    memory_size=500
).to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

# compatibil cu ambele formate
if isinstance(checkpoint, dict) and \
   "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
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
    "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test"
)

all_scores = []
all_labels = []

num_frames = 0

start_time = time.time()

for test_dir in sorted(
    test_root.glob("Test0*")
):

    if test_dir.name.endswith("_gt"):
        continue

    gt_dir = Path(
        str(test_dir) + "_gt"
    )

    print(
        f"\nProcessing {test_dir.name}"
    )

    frame_paths = sorted(
        test_dir.glob("*.tif")
    )

    gt_paths = sorted(
        gt_dir.glob("*.bmp")
    )

    video_scores = []
    video_labels = []

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

            recon, _, _, _ = model(
                image
            )

            score = criterion(
                recon,
                image
            ).item()

        gt = Image.open(
            gt_path
        ).convert("L")

        label = (
            1
            if max(gt.getdata()) > 0
            else 0
        )

        video_scores.append(score)
        video_labels.append(label)

        all_scores.append(score)
        all_labels.append(label)

        num_frames += 1

    if len(set(video_labels)) > 1:

        video_auc = roc_auc_score(
            video_labels,
            video_scores
        )

        print(
            f"{test_dir.name} "
            f"AUC = {video_auc:.4f}"
        )

    else:

        print(
            f"{test_dir.name} "
            f"AUC = N/A"
        )

    plt.figure(figsize=(10, 4))

    plt.plot(video_scores)

    plt.title(
        f"{test_dir.name} "
        f"Reconstruction Error"
    )

    plt.xlabel("Frame")

    plt.ylabel("Error")

    plt.tight_layout()

    plt.savefig(
        f"{test_dir.name}_scores.png"
    )

    plt.close()

elapsed = (
    time.time()
    - start_time
)

fps = (
    num_frames / elapsed
)

frame_auc = roc_auc_score(
    all_labels,
    all_scores
)

print("\n====================")
print(
    f"FRAME-LEVEL AUC: "
    f"{frame_auc:.6f}"
)

print(
    f"FPS: {fps:.2f}"
)
print("====================")