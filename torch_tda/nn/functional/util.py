import torch
import numpy as np

def dgms_tensor_list(ReducedCC, maxHomdim):
    """
    Persistent Diagrams to tensor list
    Return a list of tensors, where each tensor is a Persistent Diagrams at a dimension
    Note:
    1. We also return zero length Bar for gradient computation consideration
    2. The everlasting persistence bars will have death index as 
        64-bit unsigned maximum integer, and so we set it to be -1 here. 
    3. The everlasting persistence bars will have death values as float('inf'), 
        so you can use `p.death() == float('inf')` to find the it.
        
    -Inputs:
        ReducedCC - Reduced Chain Complex in bats
        maxHomdim - maximum homology dimension

    -Outputs:
        dgms: a list of PD tensors
        bdinds: birth-death indices (used to find gradient later)
    """
    dgms = []
    bdinds = []
    for i in range(maxHomdim + 1):
        bd_pair, bd_inds = ReducedCC.persistence_pairs_vec(i)
        # bd_pair: a list of birth-death pairs with 
        # length 2 * number of bd pairs. 
        # also note that it includes zeros length ones
        bd_inds = np.array(bd_inds)
        bd_pair = np.array(bd_pair)
        bd_inds[bd_inds == 0xFFFFFFFFFFFFFFFF] = -1 # take care of bats.NO_IND

        # convert to tensor
        bd_pair = torch.tensor(bd_pair.reshape(-1,2), requires_grad = True)
        bd_inds = torch.tensor(bd_inds.reshape(-1,2), requires_grad = False, dtype=torch.long)

        # add it
        dgms.append(bd_pair)
        bdinds.append(bd_inds)

    return dgms, bdinds
