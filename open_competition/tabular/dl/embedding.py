import torch
import torch.nn as nn
import numpy as np


class EmbeddingFactory(nn.Module):
    def __init__(self, x, dim_out):
        super().__init__()
        self.dim_out = dim_out
        self.module_list = nn.ModuleList(
            [nn.Embedding(len(set(np.unique(x[:, col]))), dim_out) for col in range(x.shape[1])])

    def forward(self, x):
        result = [self.module_list[col](x[:, col]).unsqueeze(2) for col in range(x.shape[1])]
        return torch.cat(result, dim=2)