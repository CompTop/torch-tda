import time
import torch
import torch.nn as nn
import torch_tda
import numpy as np
import matplotlib.pyplot as plt
import bats
# from tqdm import tqdm # see optimization process

# bats flags for reduction options
flags = (bats.standard_reduction_flag(), bats.clearing_flag())
# We optimze H1 and H0 and using clearing with Cohomology by setting degree = +1
layer = torch_tda.nn.RipsLayer(maxdim = 1, degree = +1 ,reduction_flags=flags) 

# generate datasets
n = 100
np.random.seed(0)
data = np.random.uniform(0,1,(n,2))
X = torch.tensor(data, requires_grad=True)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(data[:,0], data[:,1])

# objective function
f1 = torch_tda.nn.BarcodePolyFeature(1,2,0)
# optimizer 
optimizer = torch.optim.Adam([X], lr=1e-2)

# run for 10 times
for i in range(10):
    optimizer.zero_grad()
    dgms = layer(X)
    loss = -f1(dgms) 
    loss.backward()
    optimizer.step()

print("run succeffully!")

plt.subplot(1, 2, 2)
plt.scatter(X.detach().numpy()[:,0], X.detach().numpy()[:,1])
plt.savefig('opt_result.pdf', dpi=200, bbox_inches='tight')