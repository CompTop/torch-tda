# The following code is based on code originally obtained from the topology layer
# pacakge (https://github.com/bruel-gabrielsson/TopologyLayer) which is issued
# under the following license
#
#
# MIT License
#
# Copyright (c) 2019 Rickard Brüel-Gabrielsson, Bradley J. Nelson, Anjan Dwaraknath, Primoz Skraba
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import numpy as np

def remove_zero_bars(dgm):
    """
    remove zero bars from diagram
    """
    inds = dgm[:,0] != dgm[:,1]
    return dgm[inds,:]


def remove_infinite_bars(dgm, issub):
    """
    remove infinite bars from diagram
    """
    if issub:
        inds = dgm[:, 1] != np.inf
        return dgm[inds,:]
    else:
        inds = dgm[:, 1] != -np.inf
        return dgm[inds,:]

def get_start_end(dgm, issublevel):
    """
    get start and endpoints of barcode pairs
    input:
        dgminfo - Tuple consisting of diagram tensor and bool
            bool = true if diagram is of sub-level set type
            bool = false if diagram is of super-level set type
    output - start, end tensors of diagram
    """
    if issublevel:
        # sub-level set filtration e.g. Rips
        start, end = dgm[:,0], dgm[:,1]
    else:
        # super-level set filtration
        end, start = dgm[:,0], dgm[:,1]
    return start, end


def get_raw_barcode_lengths(dgm, issublevel):
    """
    get barcode lengths from barcode pairs
    no filtering
    """
    start, end = get_start_end(dgm, issublevel)
    lengths = end - start
    return lengths


def get_barcode_lengths(dgm, issublevel):
    """
    get barcode lengths from barcode pairs
    filter out infinite bars
    """
    lengths = get_raw_barcode_lengths(dgm, issublevel)
    # remove infinite and irrelevant bars
    lengths[lengths == np.inf] = 0
    lengths[lengths != lengths] = 0
    return lengths


class SumBarcodeLengths(nn.Module):
    """
    Layer that sums up lengths of barcode in persistence diagram
    ignores infinite bars, and padding
    Options:
        dim - bardocde dimension to sum over (defualt 0)
    forward input:
        (dgms, issub) tuple, passed from diagram layer
    """
    def __init__(self, dim=0):
        super(SumBarcodeLengths, self).__init__()
        self.dim=dim

    def forward(self, dgminfo):
        dgms, issublevel = dgminfo
        lengths = get_barcode_lengths(dgms[self.dim], issublevel)

        # return the sum of the barcode lengths
        return torch.sum(lengths, dim=0)


def get_barcode_lengths_means(dgm, issublevel):
    """
    return lengths and means of barcode
    set irrelevant or infinite to zero
    """
    start, end = get_start_end(dgm, issublevel)
    lengths = end - start
    means = (end + start)/2
    # remove infinite and irrelvant bars
    means[lengths == np.inf] = 0 # note this depends on lengths
    means[lengths != lengths] = 0
    lengths[lengths == np.inf] = 0
    lengths[lengths != lengths] = 0
    return lengths, means


class BarcodePolyFeature(nn.Module):
    """
    applies function
    sum length^p * mean^q
    over lengths and means of barcode
    parameters:
        p - exponent for lengths
        q - exponent for means
        remove_zero = Flag to remove zero-length bars (default=True)
    """
    def __init__(self, p, q, remove_zero=True):
        super(BarcodePolyFeature, self).__init__()
        # self.dim = dim
        self.p = p
        self.q = q
        self.remove_zero = remove_zero

    def forward(self, dgm, issublevel = True):
        # dgm: torch tensor of shape (n,2) 
        # dgm = dgms[self.dim]
        if self.remove_zero:
            dgm = remove_zero_bars(dgm)
        lengths, means = get_barcode_lengths_means(dgm, issublevel)

        return torch.sum(torch.mul(torch.pow(lengths, self.p), torch.pow(means, self.q)))


def pad_k(t, k, pad=0.0):
    """
    zero pad tensor t until dimension along axis is k
    if t has dimension greater than k, truncate
    """
    lt = len(t)
    if lt > k:
        return t[:k]
    if lt < k:
        fillt = torch.tensor(pad * np.ones(k - lt), dtype=t.dtype)
        return torch.cat((t, fillt))
    return t


class TopKBarcodeLengths(nn.Module):
    """
    Layer that returns top k lengths of persistence diagram in dimension
    inputs:
        dim - homology dimension
        k - number of lengths
    ignores infinite bars and padding
    """
    def __init__(self, dim, k):
        super(TopKBarcodeLengths, self).__init__()
        self.k = k
        self.dim = dim

    def forward(self, dgminfo):
        dgms, issublevel = dgminfo
        lengths = get_barcode_lengths(dgms[self.dim], issublevel)

        # sort lengths
        sortl, indl = torch.sort(lengths, descending=True)

        return pad_k(sortl, self.k, 0.0)


class PartialSumBarcodeLengths(nn.Module):
    """
    Layer that computes a partial sum of barckode lengths
    inputs:
        dim - homology dimension
        skip - skip this number of the longest bars
    ignores infinite bars and padding
    """
    def __init__(self, dim, skip):
        super(PartialSumBarcodeLengths, self).__init__()
        self.skip = skip
        self.dim = dim

    def forward(self, dgminfo):
        dgms, issublevel = dgminfo
        lengths = get_barcode_lengths(dgms[self.dim], issublevel)

        # sort lengths
        sortl, indl = torch.sort(lengths, descending=True)

        return torch.sum(sortl[self.skip:])
