import time
import torch
import torch.nn as nn
import torch_tda
import numpy as np
import matplotlib.pyplot as plt
import bats
from tqdm import tqdm # see optimization process

# bats flags for reduction options
flags = (bats.standard_reduction_flag(),bats.compression_flag())
layer = torch_tda.nn.RipsLayer(maxdim = 1, reduction_flags=flags) 

# generate datasets
n = 100
np.random.seed(0)
data = np.random.uniform(0,1,(n,2))
X = torch.tensor(data, requires_grad=True)

# objective function
f1 = torch_tda.nn.BarcodePolyFeature(1,2,0)
# optimizer 
optimizer = torch.optim.Adam([X], lr=1e-2)

# run for 10 times
for i in tqdm(range(10)):
    optimizer.zero_grad()
    dgms = layer(X)
    loss = -f1(dgms) 
    loss.backward()
    optimizer.step()