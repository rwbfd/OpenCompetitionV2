import torch
import torch.nn as nn
import numpy as np
from torch.util.data import Dataset

class EmbeddingFactory(nn.Module):
    def __init__(self, x, dim_out):
        super().__init__()
        self.dim_out = dim_out
        self.module_list = nn.ModuleList(
            [nn.Embedding(len(set(np.unique(x[:, col]))), dim_out) for col in range(x.shape[1])])

    def forward(self, x):
        result = [self.module_list[col](x[:, col]).unsqueeze(2) for col in range(x.shape[1])]
        return torch.cat(result, dim=2)


class TabDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = torch.from_numpy(x).type(torch.int64)
        self.y = torch.from_numpy(y).type(torch.float32).squeeze()

    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx]

    def __len__(self):
        return self.x.shape[0]