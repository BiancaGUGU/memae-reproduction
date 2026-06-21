import csv
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    DataLoader,
    random_split
)

from datasets.ucsd_dataset import (
    UCSDPed2Dataset
)

from models.conv_memae_extended import (
    ConvMemAEExtended
)

MEMORY_SIZES = [
    500,
    1000,
    1200,
    1500
]

EPOCHS = 120

BATCH_SIZE = 8

LR = 1e-4

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

dataset = UCSDPed2Dataset(
    "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train"
)

train_size = int(
    0.8 * len(dataset)
)

val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

results = []

for memory_size in MEMORY_SIZES:

    print(
        f"\n========== MEMORY {memory_size} =========="
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = ConvMemAEExtended(
        memory_size=memory_size
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR
    )

    best_val_loss = float("inf")
    best_epoch = 0

    csv_file = open(
        f"log_memory_{memory_size}.csv",
        "w",
        newline=""
    )

    writer = csv.writer(csv_file)

    writer.writerow([
        "epoch",
        "train_loss",
        "val_loss"
    ])

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0

        for images in train_loader:

            images = images.to(device)

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
                torch.log(att + 1e-12)
            ).sum(
                dim=1
            ).mean()

            compactness_loss = (
                (z - z_mem) ** 2
            ).mean()

            loss = (
                recon_loss
                + 0.0002 * entropy_loss
                + 0.01 * compactness_loss
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()

        val_loss = 0

        with torch.no_grad():

            for images in val_loader:

                images = images.to(device)

                (
                    recon,
                    _,
                    _,
                    _
                ) = model(images)

                loss = criterion(
                    recon,
                    images
                )

                val_loss += loss.item()

        val_loss /= len(val_loader)

        writer.writerow([
            epoch + 1,
            train_loss,
            val_loss
        ])

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch": best_epoch,
                    "val_loss": best_val_loss,
                    "memory_size": memory_size,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict()
                },
                f"best_checkpoint_{memory_size}.pth"
            )

        print(
            f"Memory={memory_size}"
            f" Epoch={epoch+1}/{EPOCHS}"
            f" Train={train_loss:.6f}"
            f" Val={val_loss:.6f}"
        )

    csv_file.close()

    results.append(
        (
            memory_size,
            best_epoch,
            best_val_loss
        )
    )

print("\n===== FINAL RESULTS =====")

for m, e, l in results:

    print(
        f"Memory={m}"
        f" BestEpoch={e}"
        f" BestValLoss={l:.6f}"
    )