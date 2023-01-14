
"""
Bottleneck distance

This is still under development as an open question:
Given two diagrams, it is possible that in the max edge (p1,p2) of the inf matching 
where the bottleck distance is obtained, one point p1 happens to be a point on the diagonal and it is not a 
'real' persistent pair (have a correspondence to the two birth and death simplices in the complex).

How can we deal with situation? 

My tentative solution is to find the closet point on the diagonal to p2 
to approximate the true bottleneck distance. 
"""

import torch
from torch.autograd import Function
from persim import bottleneck
import numpy as np
# import hera_tda as hera
import hera


def find_index_of_nearest(arr, pt):
    distance = np.linalg.norm(arr - pt, ord=2, axis = 1)
    return np.argmin(distance)

def seperate_zero_bars(dgm):
    """
    remove zero bars from diagram
    """
    inds = dgm[:,0] != dgm[:,1]
    return dgm[inds,:], dgm[~inds,:]
    
def bott_dist_torch(in_dgm1, in_dgm2, zero_out = False):
        # print("new hera bottleneck layer")
        if not zero_out:
            dgm1, zero_dgm1 = seperate_zero_bars(in_dgm1)
            dgm2, zero_dgm2 = seperate_zero_bars(in_dgm2)
        

        d1 = dgm1.detach().numpy()
        d2 = dgm2.detach().numpy()
        # find the bottleneck distance and the maixmum edge (two points in R^2)
        dist, edge = hera.bottleneck_dist(d1, d2, return_bottleneck_edge=True)
        
        # change the data type
        b = [edge[1].get_birth(), edge[1].get_death()]
        a = [edge[0].get_birth(), edge[0].get_death()]
        # find the index of persistence pair in origin input diagrams
        idx1 = np.where((d1 == np.array(a)).all(axis=1))
        idx2 = np.where((d2 == np.array(b)).all(axis=1))
        
        # Assume at least one point is off-diagonal as both diagonal situtation is rare
        if dgm1[idx1].shape[0] == 0: 
            # one point on the diagonal dgm1 is matched to a point in dgm2 
            if dgm2.requires_grad: # do not bother to modify dgm1
                return (dgm2[idx2][0][1] - dgm2[idx2][0][0])/2
            else: # now we do not have 
                # print("undefined matching")
                closet_idx = find_index_of_nearest(zero_dgm1.detach().numpy(), d2[idx2][0])
                # need to find the closet pt in Diag of dgm1
                return torch.max(torch.abs(dgm2[idx2] -  zero_dgm1[closet_idx]))
        elif dgm2[idx2].shape[0] == 0:
            if dgm1.requires_grad:
                return (dgm1[idx1][1] - dgm1[idx1][0])/2
            else:
                # print("undefined matching")
                closet_idx = find_index_of_nearest(zero_dgm2.detach().numpy(), d1[idx1][0])
                # need to find the closet pt in Diag of dgm1
                return torch.max(torch.abs(dgm1[idx1] -  zero_dgm2[closet_idx])), dist
        else:
            return torch.max(torch.abs(dgm1[idx1] -  dgm2[idx2]))



class BottleneckDistanceHera(Function):
    """
    Compute bottleneck distance between two persistence diagrams

    TODO: This torch function is problematic, should use the above bott_dist_torch() function


    forward inputs:
        dgm1 - N x 2 torch.float tensor of birth-death pairs
        dgm2 - M x 2 torch.float tensor of birth-death pairs
    """

    @staticmethod
    def forward(ctx, dgm1, dgm2):
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




class BottleneckDistance(Function):
    """
    Compute bottleneck distance between two persistence diagrams by persim

    forward inputs:
        dgm0 - N x 2 torch.float tensor of birth-death pairs
        dgm1 - M x 2 torch.float tensor of birth-death pairs
    """
    @staticmethod
    def forward(ctx, dgm0, dgm1):
        ctx.dtype = dgm0.dtype
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

        d01 = torch.tensor(d0[i0] - d1[i1], dtype=ctx.dtype)
        ctx.d01 = d01
        dist01 = np.linalg.norm(d0[i0] - d1[i1], np.inf)
        ctx.indmax = np.argmax(np.abs(d0[i0] - d1[i1]))

        return torch.tensor(dist01, dtype=ctx.dtype)

    @staticmethod
    def backward(ctx, grad_dist):
        n0 = ctx.n0
        n1 = ctx.n1
        i0 = ctx.i0
        i1 = ctx.i1
        d01 = ctx.d01

        gd0 = torch.zeros(n0, 2, dtype=ctx.dtype)
        gd1 = torch.zeros(n1, 2, dtype=ctx.dtype)


        gd0[i0, ctx.indmax] = np.sign(d01[ctx.indmax]) * grad_dist
        gd1[i1, ctx.indmax] = -np.sign(d01[ctx.indmax]) * grad_dist

        return gd0, gd1
