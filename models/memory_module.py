import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryModule(nn.Module):

    def __init__(
        self,
        memory_size=2000,
        feature_dim=256,
        shrink_thres=0.0025
    ):
        super().__init__()

        self.shrink_thres = shrink_thres

        self.memory = nn.Parameter(
            torch.randn(
                memory_size,
                feature_dim
            )
        )

        nn.init.xavier_uniform_(
            self.memory
        )

    def hard_shrink_relu(self, x):

        return (
            F.relu(
                x - self.shrink_thres
            )
            * x
            /
            (
                torch.abs(
                    x - self.shrink_thres
                )
                + 1e-12
            )
        )

    def forward(self, z):

        z_norm = F.normalize(
            z,
            dim=1
        )

        mem_norm = F.normalize(
            self.memory,
            dim=1
        )

        att = torch.matmul(
            z_norm,
            mem_norm.t()
        )

        att = F.softmax(
            att,
            dim=1
        )

        att = self.hard_shrink_relu(
            att
        )

        att = att / (
            att.sum(
                dim=1,
                keepdim=True
            )
            + 1e-12
        )

        z_hat = torch.matmul(
            att,
            self.memory
        )

        return z_hat, att