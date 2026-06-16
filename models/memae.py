from models.autoencoder import Encoder
from models.autoencoder import Decoder
from models.memory_module import MemoryModule

import torch.nn as nn


class MemAE(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = Encoder()

        self.memory = MemoryModule(
            memory_size=2000,
            feature_dim=64
        )

        self.decoder = Decoder()

    def forward(self, x):

        z = self.encoder(x)

        z_hat, att = self.memory(z)

        recon = self.decoder(z_hat)

        return recon, att