from .functional import RipsDiagram, Rips0Diagram

import torch.nn as nn

class RipsLayer(nn.Module):
    """
    Define a Rips persistence layer that will use the Rips Diagram function
    Inpute:
        maxdim : maximum homology dimension (default=0)
        reduction_flags : PH computation options from bats
            see details in:
            https://bats-tda.readthedocs.io/en/latest/tutorials/Rips.html#Algorithm-optimization
    """
    def __init__(self, maxdim = 0, reduction_flags=()):
        super(RipsLayer, self).__init__()
        self.maxdim = maxdim
        self.PD = RipsDiagram()
        self.reduction_flags = reduction_flags

    def forward(self, x):
        xnp = x.cpu().detach().numpy() # convert to numpy array
        dgms = self.PD.apply(x, self.maxdim, *self.reduction_flags)
        return dgms


class Rips0Layer(nn.Module):
    """
    Define a Rips persistence layer that will use the Rips Diagram function
    Only computes dimension 0 using Union-Find
    """
    def __init__(self):
        super(Rips0Layer, self).__init__()
        self.PD = Rips0Diagram()

    def forward(self, x):
        xnp = x.cpu().detach().numpy() # convert to numpy array
        dgms = self.PD.apply(x)
        return dgms
