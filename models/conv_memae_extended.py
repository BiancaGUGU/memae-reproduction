import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvMemAEExtended(nn.Module):

    def __init__(
        self,
        memory_size=500,
        feature_dim=256
    ):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1, 32,
                4, 2, 1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32, 64,
                4, 2, 1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64, 128,
                4, 2, 1
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
                4, 2, 1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 32,
                4, 2, 1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                32, 1,
                4, 2, 1
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

        z_mem = torch.matmul(
            att,
            self.memory
        )

        dec = self.fc_dec(
            z_mem
        )

        dec = dec.view(
            -1,
            128,
            16,
            16
        )

        recon = self.decoder(
            dec
        )

        return (
            recon,
            att,
            z,
            z_mem
        )