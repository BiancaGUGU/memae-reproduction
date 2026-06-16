import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvMemAE(nn.Module):

    def __init__(self,
                 memory_size=200,
                 feature_dim=256):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU()
        )

        self.fc_enc = nn.Linear(
            128 * 16 * 16,
            feature_dim
        )

        self.memory = nn.Parameter(
            torch.randn(
                memory_size,
                feature_dim
            )
        )

        self.fc_dec = nn.Linear(
            feature_dim,
            128 * 16 * 16
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                32, 1,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        z = self.encoder(x)

        z = z.view(
            z.size(0),
            -1
        )

        z = self.fc_enc(z)

        att = F.softmax(
            torch.matmul(
                z,
                self.memory.t()
            ),
            dim=1
        )

        z_hat = torch.matmul(
            att,
            self.memory
        )

        z_hat = self.fc_dec(
            z_hat
        )

        z_hat = z_hat.view(
            -1,
            128,
            16,
            16
        )

        recon = self.decoder(
            z_hat
        )

        return recon, att