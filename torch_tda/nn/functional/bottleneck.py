# Bottleneck distance

import bats
import torch
from torch.autograd import Function
from persim import bottleneck
import numpy as np

class BottleneckDistance(Function):
    """
    Compute bottleneck distance between two persistence diagrams

    forward inputs:
        dgm0 - N x 2 torch.float tensor of birth-death pairs
        dgm1 - M x 2 torch.float tensor of birth-death pairs
    """
    @staticmethod
    def forward(ctx, dgm0, dgm1):
        d0 = dgm0.detach().numpy()
        d1 = dgm1.detach().numpy()
        n0 = len(dgm0)
        ctx.n0 = n0
        n1 = len(dgm1)
        ctx.n1 = n1

        dist, match = bottleneck(d0, d1, matching=True)

        imax = np.argmax(match[:,2])
        i0, i1, dm = match[imax]
        i0, i1 = int(i0), int(i1)
        # TODO check for -1 as index

        ctx.i0 = i0
        ctx.i1 = i1

        d01 = torch.tensor(d0[i0] - d1[i1])
        ctx.d01 = d01
        dist01 = np.linalg.norm(d0[i0] - d1[i1], np.inf)
        ctx.indmax = np.argmax(np.abs(d0[i0] - d1[i1]))

        return torch.tensor(dist01)

    @staticmethod
    def backward(ctx, grad_dist):
        n0 = ctx.n0
        n1 = ctx.n1
        i0 = ctx.i0
        i1 = ctx.i1
        d01 = ctx.d01

        gd0 = torch.zeros(n0, 2)
        gd1 = torch.zeros(n1, 2)


        gd0[i0, ctx.indmax] = d01[ctx.indmax] * grad_dist
        gd1[i1, ctx.indmax] = -d01[ctx.indmax] * grad_dist

        return gd0, gd1
