import bats
import torch
from torch.autograd import Function
import scipy.spatial.distance as distance
from .util import dgms_tensor_list

def compute_y_gradient(X, F, R, imap, grad_dgms):
    '''
    -Input: 
        X: original data points
        F: Filtration built on X
        R: Reduced Chain Complex computed on F
        imap: inverse map that maps the birth or death index of a persistence pair 
            to the critical edge that creates or destroys it.
        grad_dgms: gradient with respect to PDs that is accumulated from the last layer.

    -Output: 
        y_grad: gradient values that is to be updated to X.
    
    '''
    y_grad = torch.zeros_like(X)
    scplex = F.complex()

    for i, grad_dgm in enumerate(grad_dgms):
        ps = R.persistence_pairs(i) # persistence pair at dim i
        
        for ind, gd in enumerate(grad_dgm):
            # non-zero gradient
            if not torch.equal(gd, torch.tensor([0.,0.], dtype=torch.double)):
                # find correponding critical simplex index in filtration
                p = ps[ind]
                d = p.dim() # homology dimension

                # Birth
                # get indices of two vertices of the (birth) edge
                bi = p.birth_ind() # index of birth edge

                [birth_vertex1, birth_vertex2] = scplex.get_simplex(1, bi)

                # compute gradient on y now for birth index
                dx = X[birth_vertex1] - X[birth_vertex2]
                if torch.norm(dx) == 0:
                    dx = 0
                else:
                    dx /= torch.norm(dx)
                y_grad[birth_vertex1] = gd[0] * dx
                y_grad[birth_vertex2] = - gd[0] * dx

                # Death (avoid infinite death)
                if p.death() != float('inf') and p.death_ind() <= len(imap[d+1])-1:
                    # p.death_ind() now is the index of an triangle that destroys H1
                    # we need to map it to the critical edge that creates it
                    di = imap[d+1][p.death_ind()]

                    # get index of two vertices of the (death) edge
                    [death_vertex1, death_vertex2] = scplex.get_simplex(1, di)

                    # compute gradient on y now for death index
                    dx = X[death_vertex1] - X[death_vertex2]
                    dx /= torch.norm(dx)
                    y_grad[death_vertex1] = gd[1] * dx
                    y_grad[death_vertex2] = - gd[1] * dx


    return y_grad


class RipsDiagram(Function):
    """
    Compute Rips complex persistence using point coordinates
    forward inputs:
        y - N x D torch.float tensor of coordinates (original data points)
        maxdim - maximum homology dimension
        reduction_flags - optional reduction flags for BATS.
    """
    @staticmethod
    def forward(ctx, y, maxdim, *reduction_flags):
        # number of arguments should match the return of backward function

        ynp = y.detach().numpy()
        DX = distance.squareform(distance.pdist(ynp))
        rX = bats.enclosing_radius(bats.Matrix(DX))
        # maixmum complex dimension = maximum homology dimension + 1
        F, imap = bats.LightRipsFiltration_extension(bats.Matrix(DX), rX , maxdim+1)
        R = bats.reduce(F, bats.F2(), *reduction_flags)

        # store device
        device = y.device
        ctx.device = device
        ycpu = y.cpu()

        # save data coordinates for backward
        ctx.save_for_backward(ycpu)
        # return persistent diagrams with death and birth values
        dgms, bdinds = dgms_tensor_list(R, maxdim)

        # use context variable `ctx` to store information that will be used for backward
        ctx.R = R
        ctx.filtration = F
        ctx.imap = imap
        ctx.reduction_flags = reduction_flags
        return tuple(dgms)

    @staticmethod
    def backward(ctx, *grad_dgms):
        """
        Input: 
            grad_dgms: a Tensor containing the gradient of the loss
            with respect to the output of this layer (i.e., gradient w.r.t. PDs) 
            and now we need to compute the gradient of the loss 
            with respect to the first argument in input (original data points).
        """

        # find returned gradient, which is the same size as dgms in forward function
        grad_dgm = [gd.cpu() for gd in grad_dgms]

        # find the gradient on y with the same shape as y
        device = ctx.device
        ycpu, = ctx.saved_tensors
        R = ctx.R
        F = ctx.filtration
        imap = ctx.imap

        grad_y = compute_y_gradient(ycpu, F, R, imap, grad_dgms)

        # backward only to the first argument in inputs: y 
        ret = [grad_y.to(device), None] 
        # 'None' here means that we need to match the forward function arguments
        # and `maxdim` and `*reduction_flags` cannot do gradient-descent
        ret.extend([None for f in ctx.reduction_flags])
        return tuple(ret)
