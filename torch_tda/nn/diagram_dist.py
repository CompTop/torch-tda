import torch.nn as nn
from .functional import BottleneckDistance, WassersteinDistance
from .poly_feat import remove_zero_bars

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
