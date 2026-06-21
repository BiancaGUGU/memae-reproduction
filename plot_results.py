import pandas as pd
import matplotlib.pyplot as plt

sizes = [
    500,
    1000,
    1200,
    1500
]

for size in sizes:

    df = pd.read_csv(
        f"log_memory_{size}.csv"
    )

    plt.figure()

    plt.plot(
        df["epoch"],
        df["train_loss"],
        label="Train"
    )

    plt.plot(
        df["epoch"],
        df["val_loss"],
        label="Validation"
    )

    plt.title(
        f"Memory {size}"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.savefig(
        f"memory_{size}.png"
    )

    plt.close()