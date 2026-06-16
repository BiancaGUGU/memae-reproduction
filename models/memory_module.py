import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryModule(nn.Module):

    def __init__(self,
                 memory_size=2000,
                 feature_dim=64):

        super().__init__()

        self.memory = nn.Parameter(
            torch.randn(memory_size, feature_dim)
        )

    # def forward(self, z):

    #     att = F.softmax(
    #         torch.matmul(z, self.memory.t()),
    #         dim=1
    #     )

    #     z_hat = torch.matmul(
    #         att,
    #         self.memory
    #     )

    #     return z_hat, att
    def forward(self, z):

        att = F.softmax(
            torch.matmul(z, self.memory.t()),
            dim=1
        )

        k = 10

        values, indices = torch.topk(
            att,
            k=k,
            dim=1
        )

        sparse_att = torch.zeros_like(att)

        sparse_att.scatter_(
            1,
            indices,
            values
        )

        sparse_att = sparse_att / (
            sparse_att.sum(
                dim=1,
                keepdim=True
            ) + 1e-12
        )

        z_hat = torch.matmul(
            sparse_att,
            self.memory
        )

        return z_hat, sparse_att