# coding = 'utf-8'
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 0.00001


class EntityEmbeddingLayer(nn.Module):
    def __init__(self, num_level, emdedding_dim, centroid):
        super(EntityEmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_level, emdedding_dim)
        self.centroid = torch.tensor(centroid).detach_().unsqueeze(1)

    def forward(self, x):
        """
        x must be batch_size times 1
        """
        x = x.unsqueeze(1)
        d = 1.0 / ((x - self.centroid).abs() + EPS)
        w = F.softmax(d.squeeze(2), 1)
        v = torch.mm(w, self.embedding.weight)
        return v

