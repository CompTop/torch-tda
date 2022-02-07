# Distance based on 2 Pesistence Diagrams:
# 1. BottleneckDistance
# 2. WassersteinDistance
import torch.nn as nn
from .functional import BottleneckDistance, WassersteinDistance, BottleneckDistanceHera
from .poly_feat import remove_zero_bars

# `from . import` is called Intra-package References 
# see in https://docs.python.org/3/tutorial/modules.html#intra-package-references

class BottleneckLayerHera(nn.Module):
    def __init__(self):
        super(BottleneckLayerHera, self).__init__()
        self.D = BottleneckDistanceHera()

    def forward(self, dgm0, dgm1):
        dgm0 = remove_zero_bars(dgm0)
        dgm1 = remove_zero_bars(dgm1)

        return self.D.apply(dgm0, dgm1)

class BottleneckLayer(nn.Module):
    def __init__(self):
        super(BottleneckLayer, self).__init__()
        self.D = BottleneckDistance()

    def forward(self, dgm0, dgm1):
        dgm0 = remove_zero_bars(dgm0)
        dgm1 = remove_zero_bars(dgm1)

        return self.D.apply(dgm0, dgm1)

class WassersteinLayer(nn.Module):
    def __init__(self):
        super(WassersteinLayer, self).__init__()
        self.D = WassersteinDistance()

    def forward(self, dgm0, dgm1):
        dgm0 = remove_zero_bars(dgm0)
        dgm1 = remove_zero_bars(dgm1)

        return self.D.apply(dgm0, dgm1)
