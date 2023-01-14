# Distance based on 2 Pesistence Diagrams:
# 1. BottleneckDistance
# 2. WassersteinDistance
import torch.nn as nn
from .functional import BottleneckDistance, WassersteinDistance, BottleneckDistanceHera
from .poly_feat import remove_zero_bars
import numpy as np
# import hera_tda as hera
import hera
import torch

# `from . import` is called Intra-package References 
# see in https://docs.python.org/3/tutorial/modules.html#intra-package-references

class BottleneckLayerHera(nn.Module):
    def __init__(self):
        super(BottleneckLayerHera, self).__init__()
        # self.D = BottleneckDistanceHera()

    def forward(self, dgm1, dgm2, zero_out = True):
        print("new hera bottleneck layer")
        if not zero_out:
            dgm1 = remove_zero_bars(dgm1)
            dgm2 = remove_zero_bars(dgm2)
        
        # return self.D.apply(dgm0, dgm1)

        d1 = dgm1.detach().numpy()
        d2 = dgm2.detach().numpy()
        # find the bottleneck distance and the maixmum edge (two points in R^2)
        dist, edge = hera.bottleneck_dist(d1, d2, return_bottleneck_edge=True)
        
        # change the data type
        b = [edge[1].get_birth(), edge[1].get_death()]
        a = [edge[0].get_birth(), edge[0].get_death()]
        # find the index of persistence pair in origin input diagrams
        idx1, idx2 = np.where((d1 == np.array(a)).all(axis=1)), np.where((d2 == np.array(b)).all(axis=1))


        if dgm1[idx1].shape[0] == 0:
            return (dgm2[idx2][0][1] - dgm2[idx2][0][0])/2
        elif dgm2[idx2].shape[0] == 0:
            return (dgm1[idx1][0][1] - dgm1[idx1][0][0])/2
        else:
            return torch.max(torch.abs(dgm1[idx1] -  dgm2[idx2]))


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
