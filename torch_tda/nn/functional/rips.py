import bats
import torch
from torch.autograd import Function
import scipy.spatial.distance as distance
from .util import dgms_tensor_list

def compute_y_gradient(X, F, R, imap, grad_dgms):
    '''
    TODO: torch.norm(dx) might be zero
    '''
    y_grad = torch.zeros_like(X)
    scplex = F.complex()

    for i in range(len(grad_dgms)):
        grad_dgm = grad_dgms[i]
        ind_bar = 0 # find non-zero gradient's index (bar index)
        ps = R.persistence_pairs(i)

        for gd in grad_dgm:
            # non-zero gradient
            if not torch.equal(gd, torch.tensor([0.,0.])):
                # find correponding critical simplex index in filtration
                p = ps[ind_bar]
                d = p.dim() # homology dimension

                # Birth
                # get indices of two vertices of the (birth) edge
                bi = p.birth_ind() # index of birth edge

                [birth_vertex1, birth_vertex2] = scplex.get_simplex(1, bi)

                # compute gradient on y now for birth index
                dx = X[birth_vertex1] - X[birth_vertex2]
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


            ind_bar+=1

    return y_grad


class RipsDiagram(Function):
    """
    Compute Rips complex persistence using point coordinates
    forward inputs:
        y - N x D torch.float tensor of coordinates
        maxdim - maximum homology dimension
        alg - algorithm
            'hom' = homology (default)
            'hom2' = nz suppressing homology variant
            'cohom' = cohomology
    """
    @staticmethod
    def forward(ctx, y, maxdim):
        # number of arguments should match the return of backward function

        ynp = y.detach().numpy()
        DX = distance.squareform(distance.pdist(ynp))
        rX = bats.enclosing_radius(bats.Matrix(DX))
        # maixmum complex dimension = maximum homology dimension + 1
        F, imap = bats.LightRipsFiltration_extension(bats.Matrix(DX), rX , maxdim+1)
        R = bats.reduce(F, bats.F2())

        # store device
        device = y.device
        ctx.device = device
        ycpu = y.cpu()

        # save data coordinates for backward
        ctx.save_for_backward(ycpu)
        # return persistent diagrams with death and birth values
        dgms = dgms_tensor_list(R, maxdim)


        ctx.R = R
        ctx.filtration = F
        ctx.imap = imap
        #print("in RipsDiagram.forward")
        #print(dgms)
        return tuple(dgms)

    @staticmethod
    def backward(ctx, *grad_dgms):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output of this layer, and we need to compute the gradient of the loss
        with respect to the input.
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

        # backward only to inputs of coordinates of points
        # grad_y should be timed together with the last layer
        return grad_y.to(device), None
