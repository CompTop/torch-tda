import torch
from torch.autograd import Function
from .util import dgms_tensor_list
import bats

class SublevelsetPersistence(Function):
    """
    Compute persistent homology of a

    forward inputs:
        X - a bats SimplicialComplex or LightSimplicialComplex
        f - funcion on the vertex set
    """
    @staticmethod
    def forward(ctx, f, X, maxdim, *reduction_flags):
        ctx.dtype = f.dtype
        f = f.detach().numpy()
        ctx.shape = f.shape

        vals, imap = bats.lower_star_filtration(X, f)
        ctx.imap = imap
        F = bats.Filtration(X, vals)
        R = bats.reduce(F, bats.F2(), *reduction_flags)
        dgms, ctx.binds = dgms_tensor_list(R, maxdim)
        ctx.reduction_flags = reduction_flags

        return tuple(dgms)


    @staticmethod
    def backward(ctx, *grad_dgm):
        device = grad_dgm[0].device

        grad_dgm = [gd.detach().tolist() for gd in grad_dgm]
        bdinds = [bd.detach().tolist() for bd in ctx.binds]
        grad_f = bats.lower_star_backwards(grad_dgm, bdinds, ctx.imap)
        grad_f = torch.tensor(grad_f, dtype=ctx.dtype)

        ret = [grad_f.to(device), None, None]
        ret.extend([None for f in ctx.reduction_flags])

        return tuple(ret)
