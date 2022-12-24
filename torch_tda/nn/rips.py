from .functional import RipsDiagram, Rips0Diagram, sparse_pairwise_dist

from sklearn.metrics.pairwise import pairwise_distances
import torch.nn as nn
import bats
import numpy as np
import torch 
def RCC_to_persistence_vertex_indices(R, scplex, imap):
    '''
    R: ReducedFilteredF2DGVectorSpace
    scplex: filtration's complex, eg., F.complex()
    imap (list of length complex's highest dimension): 
        inverse map that is able to map the index of simplex to its edge index 

    '''        
    def find_vet_idx(a):
        """Find indices vertices of a given edge index, a of shape (1,)  """
        return scplex.get_simplex(1, int(a[0]))

    reg_pss = [] 
    ess_pss = []
    for d in range(R.maxdim()):
        # find birth death persistence pairs and their corresponding simplex indices 
        bd_pair, bd_inds = R.persistence_pairs_vec(d) # we DO Not use bd_pair for auto diff
        bd_inds = np.array(bd_inds, dtype = np.uint64).reshape(-1,2)
        if d == 0:
            idxes = bd_inds[:,1] == 0xFFFFFFFFFFFFFFFF # Infinite(essential) pairs' death index
            ess_ps0 = bd_inds[idxes][:,0] # essential pairs' birth index shape (n,)
            x = bd_inds[~idxes] # regular bd indices
            reg_ps0 = np.hstack((x[:,0:1], np.apply_along_axis(find_vet_idx, 1, x[:,1:2]))) # shape (n, 3)

        else:
            # find regular and essential pairs' indices
            idxes = bd_inds[:,1] == 0xFFFFFFFFFFFFFFFF # Infinite(essential) pairs' death index
            # essential pairs 
            x = bd_inds[idxes][:,0:1] # locate the index of each birth simplex at dimension d
            # x.shape == (n, 1)
            if x.shape[0] == 0: # empty array
                ess_ps = np.array([])
            else:
                # inverse map birth simplex indices to their edge indices
                x = np.apply_along_axis(lambda a: imap[d][a[0]], 1, x).reshape(-1,1)
                # find the vertex indices of each birth edge with shape (n,2)
                ess_ps = np.apply_along_axis(find_vet_idx, 1, x) 
            
            # regular pairs 
            x = bd_inds[~idxes] # locate the index of each birth simplex and death simplex
            # x.shape == (n, 2)
            edge_births = np.apply_along_axis(lambda a: imap[d][a[0]] , 1, x[:,0:1]).reshape(-1,1)
            edge_deaths = np.apply_along_axis(lambda a: imap[d+1][a[0]] , 1, x[:,1:2]).reshape(-1,1)
            
            reg_ps = np.hstack((np.apply_along_axis(find_vet_idx, 1, edge_births), 
                                np.apply_along_axis(find_vet_idx, 1, edge_deaths)))
            
            # add it
            reg_pss.append(reg_ps)
            ess_pss.append(ess_ps) 
    return reg_ps0, reg_pss, ess_ps0, ess_pss



class RipsLayer(nn.Module):
    """
    Define a Rips persistence layer that will use the Rips Diagram function.
    Here we return the all essential and regular persistence pairs
        we leave users to decide if they want to use essential pairs or zero-length bars
        in practice   
    Input:
        maxdim : maximum homology dimension (default=0)
        degree : +1 Cohomology; -1 Homology
        sparse : True if you want to use sparse Rips
        eps    : approximation error bound of bottleneck distance for sparse Rips
        metric : scikit-learn use to compute sklearn.metrics.pairwise.pairwise_distances
        reduction_flags : PH computation options from bats
            see details in:
            https://bats-tda.readthedocs.io/en/latest/tutorials/Rips.html#Algorithm-optimization
    
    Output:
        dgms    : list of length `maxdim`, where each element is an numpy array of shape (n,2)
                note: infinite death == float('inf')
        bdinds  : list of length `maxdim`, where each element is an numpy array of shape (n,2)
                note: infinite death index == -1
    """
    def __init__(self, maxdim = 0, degree = -1, metric = 'euclidean', sparse = False, eps=0.5,  reduction_flags=()):
        super(RipsLayer, self).__init__()
        self.maxdim = maxdim
        self.degree = degree
        self.sparse = sparse
        self.eps = eps
        # self.PD = RipsDiagram()
        self.metric = metric
        self.reduction_flags = reduction_flags

    def forward(self, X):
        # dgms = self.PD.apply(x, self.maxdim, self.degree, self.metric , self.sparse, self.eps, *self.reduction_flags)
        # change dgms to make it able auto-diff
        Xnp = X.cpu().detach().numpy() # convert to numpy array
        # Xnp.astype('double')
        if self.sparse:
            D = pairwise_distances(Xnp, metric=self.metric)
            DX, rX = sparse_pairwise_dist(D, self.eps, dense_output = False).astype('double')
            print("sparse Rips is under-developing now and not nessarily efficient than normal rips")
        else:
            # DX = distance.squareform(distance.pdist(ynp))
            DX = pairwise_distances(Xnp, metric=self.metric).astype('double')
            rX = bats.enclosing_radius(bats.Matrix(DX))
        
        # maixmum complex dimension = maximum homology dimension + 1
        F, imap = bats.LightRipsFiltration_extension(bats.Matrix(DX), rX , self.maxdim+1)
        FVS = bats.FilteredF2DGVectorSpace(F, self.degree)
        R = bats.ReducedFilteredF2DGVectorSpace(FVS, *self.reduction_flags)
        reg_ps0, reg_pss, ess_ps0, ess_pss = RCC_to_persistence_vertex_indices(R, F.complex(), imap)

        # find persistence diagrams from persistence indices (in the format of vertex indices)
        persistence_dgs = []
        # zero dimension regular persistence diagram 
        dgm0_reg = torch.hstack((torch.zeros(reg_ps0.shape[0], 1), 
                    torch.norm(X[reg_ps0[:, 1]] - X[reg_ps0[:,2]], dim=-1).reshape(-1,1))
                    ) 
        persistence_dgs.append(dgm0_reg)
        # other dimension regular persistence diagrams
        dgms_reg = []
        for inds in reg_pss:
            dgms_reg.append(torch.norm(X[inds[:, (0, 2)]] - X[inds[:, (1, 3)]], dim=-1))
        persistence_dgs.append(dgms_reg)
        # zero dimension essential persistence diagram
        persistence_dgs.append(torch.zeros(ess_ps0.shape[0], 1))
        # other dimension essential persistence diagram
        dgms_ess = []
        for inds in ess_pss:
            if inds.shape[0] == 0:
                dgms_ess.append(torch.tensor([]))
            else:
                dgms_ess.append(torch.norm(X[inds[:, 1]] - X[inds[:, 0]], dim=-1))
        persistence_dgs.append(dgms_ess)

        return persistence_dgs


class Rips0Layer(nn.Module):
    """
    Define a Rips persistence layer that will use the Rips Diagram function
    Only computes dimension 0 using Union-Find
    """
    def __init__(self, metric = 'euclidean'):
        super(Rips0Layer, self).__init__()
        self.metric = metric
        self.PD = Rips0Diagram()

    def forward(self, x):
        xnp = x.cpu().detach().numpy() # convert to numpy array
        dgms = self.PD.apply(x, self.metric)
        return dgms
