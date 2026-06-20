from pathlib import Path

from PIL import Image

from torch.utils.data import Dataset

from torchvision import transforms


class UCSDPed2Dataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = Path(root_dir)

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        self.frame_paths = []

        for video_dir in sorted(
            self.root_dir.iterdir()
        ):

            if not video_dir.is_dir():
                continue

            for frame in sorted(
                video_dir.glob("*.tif")
            ):

                try:

                    Image.open(
                        frame
                    ).verify()

                    self.frame_paths.append(
                        frame
                    )

                except Exception:

                    print(
                        f"Corrupted file removed: {frame}"
                    )

    def __len__(self):

        return len(
            self.frame_paths
        )

    def __getitem__(self, idx):

        image = Image.open(
            self.frame_paths[idx]
        )

        image = self.transform(
            image
        )

        return image