from pathlib import Path

from PIL import Image

import torch
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

            if video_dir.is_dir():

                frames = sorted(
                    video_dir.glob("*.tif")
                )

                self.frame_paths.extend(frames)

    def __len__(self):

        return len(self.frame_paths)

    # def __getitem__(self, idx):

    #     image_path = self.frame_paths[idx]

    #     image = Image.open(
    #         image_path
    #     )

    #     image = self.transform(
    #         image
    #     )
    def __getitem__(self, idx):

        while True:

            image_path = self.frame_paths[idx]

            try:

                image = Image.open(image_path)

                image = self.transform(image)

                return image

            except Exception:

                print(
                    f"Skipping corrupted file: {image_path}"
                )

                idx = (
                    idx + 1
                ) % len(self.frame_paths)
        # return image