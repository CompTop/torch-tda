import bats
import torch
from torch.autograd import Function
import scipy.spatial.distance as distance
from .util import dgms_tensor_list
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import time
from scipy import sparse

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

def getGreedyPerm(D):
    """
    A Naive O(N^2) algorithm to do furthest points sampling

    Parameters
    ----------
    D : ndarray (N, N)
        An NxN distance matrix for points

    Return
    ------
    lamdas: list
        Insertion radii of all points
    """

    N = D.shape[0]
    #By default, takes the first point in the permutation to be the
    #first point in the point cloud, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return lambdas[perm]

def getApproxSparseDM(lambdas, eps, D):
    """
    Purpose: To return the sparse edge list with the warped distances, sorted by weight

    Parameters
    ----------
    lambdas: list
        insertion radii for points
    eps: float
        epsilon approximation constant
    D: ndarray
        NxN distance matrix, okay to modify because last time it's used

    Return
    ------
    DSparse: scipy.sparse
        A sparse NxN matrix with the reweighted edges
    """
    N = D.shape[0]
    E0 = (1+eps)/eps
    E1 = (1+eps)**2/eps

    # Create initial sparse list candidates (Lemma 6)
    # Search neighborhoods
    nBounds = ((eps**2+3*eps+2)/eps)*lambdas

    # Set all distances outside of search neighborhood to infinity
    D[D > nBounds[:, None]] = np.inf
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    idx = I < J
    I = I[(D < np.inf)*(idx == 1)]
    J = J[(D < np.inf)*(idx == 1)]
    D = D[(D < np.inf)*(idx == 1)]

    #Prune sparse list and update warped edge lengths (Algorithm 3 pg. 14)
    minlam = np.minimum(lambdas[I], lambdas[J])
    maxlam = np.maximum(lambdas[I], lambdas[J])

    # Rule out edges between vertices whose balls stop growing before they touch
    # or where one of them would have been deleted.  M stores which of these
    # happens first
    M = np.minimum((E0 + E1)*minlam, E0*(minlam + maxlam))

    t = np.arange(len(I))
    t = t[D <= M]
    (I, J, D) = (I[t], J[t], D[t])
    minlam = minlam[t]
    maxlam = maxlam[t]

    # Now figure out the metric of the edges that are actually added
    t = np.ones(len(I))

    # If cones haven't turned into cylinders, metric is unchanged
    t[D <= 2*minlam*E0] = 0

    # Otherwise, if they meet before the M condition above, the metric is warped
    D[t == 1] = 2.0*(D[t == 1] - minlam[t == 1]*E0) # Multiply by 2 convention
    return sparse.coo_matrix((D, (I, J)), shape=(N, N)).tocsr()

def sparse_pairwise_dist(D, eps = 0.15, dense_output=False):
    '''
    Generate Sparse pairwise distance matrix. In principal, the idea is to
    1) rule out of egdes that will never be added in Sparse Rips
    2) modify edges' length that will become 'cylinder' as described in paper
    
    Note:
    As eps increase, the compression rate will increase.
    
    Input
    ------
    D: N * N numpy arrary 
        pairwise distance matrix
    
    eps: float
        perisitent diagrams approximation bound
    '''
    
    t0 = time.monotonic()
    # First do furthest point sampling
    lambdas = getGreedyPerm(D)
    # Now compute the sparse distance matrix
    DSparse = getApproxSparseDM(lambdas, eps, D)
    t1 = time.monotonic()
    if dense_output:
        print("sparse Rips construction takes {} sec.".format(t1 - t0))
        print("{} left with edges".format(DSparse.nnz))
        print("{:.2f}% edges compression".format((1 - DSparse.nnz / np.prod(D.shape))*100))
    
    # return dense distance matrix
    D_dense = DSparse.todense()
    # since BATs will not consider diagonal elements when construct filtration
    # we will simply just the diagonal line to 0
    D_dense[D_dense == 0] = np.inf 
    rad = DSparse.max() + eps # radius
    return D_dense, rad

class RipsDiagram(Function):
    """
    (Outdated) we can do auto-diff by computing Diagram direcely from input point set matrix

    Compute Rips complex persistence using point coordinates
    forward inputs:
        y - N x D torch.float tensor of coordinates (original data points)
        maxdim - maximum homology dimension
        degree - +1 Cohomology; -1 Homology
        sparse - True if you want to use sparse Rips
        eps - approximation error bound of bottleneck distance for sparse Rips
        metric - options supported by scikit-learn
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
        reduction_flags - optional reduction flags for BATS.
    """
    @staticmethod
    def forward(ctx, y, maxdim, degree = -1, metric = 'euclidean', sparse=False, eps=0.5, *reduction_flags):
        # number of arguments should match the return of backward function
        ynp = y.detach().numpy()
        if sparse:
            D = pairwise_distances(ynp, metric=metric)
            DX, rX = sparse_pairwise_dist(D, eps, dense_output = False)
        else:
            # DX = distance.squareform(distance.pdist(ynp))
            DX = pairwise_distances(ynp, metric=metric)
            rX = bats.enclosing_radius(bats.Matrix(DX))
        
        # maixmum complex dimension = maximum homology dimension + 1
        F, imap = bats.LightRipsFiltration_extension(bats.Matrix(DX), rX , maxdim+1)
        FVS = bats.FilteredF2DGVectorSpace(F, degree)
        R = bats.ReducedFilteredF2DGVectorSpace(FVS, *reduction_flags)
        # R = bats.reduce(F, bats.F2(), *reduction_flags)

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
        # the length return list should match the number of inputs
        ret = [grad_y.to(device), None, None, None, None, None] 
        # 'None' here means that we need to match the forward function arguments
        # and `maxdim` and `*reduction_flags` cannot do gradient-descent
        ret.extend([None for f in ctx.reduction_flags])
        return tuple(ret)
