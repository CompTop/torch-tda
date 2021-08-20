from .functional import RipsDiagram

import torch.nn as nn

class RipsLayer(nn.Module):
    """
    Define a Rips persistence layer that will use the Rips Diagram function
    Parameters:
        maxdim : maximum homology dimension (default=0)
        alg : algorithm
            'hom' = homology (default)
            'cohom' = cohomology
    """
    def __init__(self, maxdim = 0):
        super(RipsLayer, self).__init__()
        self.maxdim = maxdim
        self.PD = RipsDiagram()

    def forward(self, x):
        xnp = x.cpu().detach().numpy() # convert to numpy array
        dgms = self.PD.apply(x, self.maxdim)
#         print("in RipsLayer")
#         print(dgms)
        return dgms
