import torch
from torch.autograd import Function
from .util import dgms_tensor_list
import bats

class SublevelsetPersistence(Function):
    """
    Compute persistent homology of a sublevelset filtration defined on X

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

        grad_f = torch.zeros(ctx.shape, dtype=ctx.dtype)
        # TODO: this is really slow
        for dim, gdgm in enumerate(grad_dgm):
            for i, gd in enumerate(gdgm):
                grad_f[ctx.imap[dim][ctx.binds[dim][i,0]]] += gd[0]
                if (not ctx.binds[dim][i][1] == -1):
                    # print(ctx.binds[dim][i,1])
                    grad_f[ctx.imap[dim+1][ctx.binds[dim][i,1]]] += gd[1]
            # TODO, this probably needs to be unpacked due to duplicate indices
            # grad_f[ctx.imap[dim][ctx.bdinds[dim][gdgm]]] += dgm

        ret = [grad_f.to(device), None, None]
        ret.extend([None for f in ctx.reduction_flags])

        return tuple(ret)
