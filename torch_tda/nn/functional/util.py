import torch

def dgms_tensor_list(ReducedCC, maxHomdim):
    """
    Persistent Diagrams to tensor list
    Return a list of tensors, where each tensor is a Persistent Diagrams at a dimension
    Note:
    1. We also return zero length Bar for gradient computation consideration
    2. The death index is 64-bit unsigned maximum integer and we set it to be -1
    3. The reason why we return
    -Inputs:
        ReducedCC - Reduced Chain Complex in bats
        maxHomdim - maximum homology dimension

    -Outputs:
        dgms: a list of PD tensors
    """
    dgms = []
    bdinds = []
    for i in range(maxHomdim + 1):
        ps = ReducedCC.persistence_pairs(i)
        bd_pair = [[p.birth(), p.death()] for p in ps]

        # convert to tensor
        bd_pair = torch.tensor(bd_pair, requires_grad = True)

        # add it
        dgms.append(bd_pair)

    return dgms
    #return dgms, bdinds
