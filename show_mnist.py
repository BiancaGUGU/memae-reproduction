import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms

dataset = MNIST(
    root="./data",
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

fig, axes = plt.subplots(2, 5)

for i in range(10):

    image, label = dataset[i]

    ax = axes[i // 5][i % 5]

    ax.imshow(
        image.squeeze(),
        cmap="gray"
    )

    ax.set_title(str(label))
    ax.axis("off")

plt.tight_layout()
plt.show()