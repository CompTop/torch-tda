from .functional import SublevelsetPersistence
import torch.nn as nn

class SublevelsetDiagram(nn.Module):
    def __init__(self, X, maxdim=0, reduction_flags=()):
        super(SublevelsetDiagram, self).__init__()
        self.X = X
        self.maxdim = maxdim
        self.reduction_flags = reduction_flags
        self.PD = SublevelsetPersistence()

    def forward(self, f):
        return self.PD.apply(f, self.X, self.maxdim, *self.reduction_flags)
