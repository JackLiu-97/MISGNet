from torch import nn
from .basicnet5 import MutualNet


class MGLNet(nn.Module):
    def __init__(self, dropout=0.1, BatchNorm=nn.BatchNorm2d, num_clusters=32, dim=32):
        super(MGLNet, self).__init__()
        self.gamma = 1.0
        self.dim = dim

        self.mutualnet0 = MutualNet(BatchNorm, dim=self.dim, num_clusters=num_clusters, dropout=dropout)

    def forward(self, x1, x2):
        x1, x2 = self.mutualnet0(x1, x2)
        return x1, x2
