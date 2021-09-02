# Wasserstein distance

import bats
import torch
from torch.autograd import Function
from persim import wasserstein
import numpy as np

class WassersteinDistance(Function):
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

        dist, match = wasserstein(d0, d1, matching=True)

        # matching indices
        ctx.match0 = np.array(match[:,0], dtype=int)
        ctx.match1 = np.array(match[:,1], dtype=int)
        ctx.offsets = np.empty((len(match), 2), dtype=float)

        # construct unit offset vector
        inds01 = ctx.match0 > -1
        ctx.offsets[inds01] = dgm0[ctx.match0[inds01]] - dgm1[ctx.match1[inds01]]

        # handle -1 in match0
        # note these are matched with dgm1
        ctx.offsets[ctx.match0 == -1] = [1, -1] # drive apart birth, death

        # handle -1 in match1
        # note these are matched with dgm0
        ctx.offsets[ctx.match1 == 1] =[-1, 1] # negative of above, drive apart birth, death

        # normalize
        ctx.offsets = ctx.offsets / np.linalg.norm(ctx.offsets, axis=1).reshape(-1,1)
        ctx.offsets = torch.tensor(ctx.offsets, dtype=torch.float)
        ctx.match0 = torch.tensor(ctx.match0, dtype=torch.long)
        ctx.match1 = torch.tensor(ctx.match1, dtype=torch.long)

        return torch.tensor(dist)

    @staticmethod
    def backward(ctx, grad_dist):

        gd0 = torch.zeros(ctx.n0, 2)
        gd1 = torch.zeros(ctx.n1, 2)

        gd0[ctx.match0] = ctx.offsets[ctx.match0] * grad_dist
        gd1[ctx.match1] = -ctx.offsets[ctx.match1] * grad_dist

        return gd0, gd1
