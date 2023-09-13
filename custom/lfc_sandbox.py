#%%

import numpy as np
import datetime

#%%

# Original code. 

# Comments added here. Author: JamieProudfoot.

#%%

# Verbose version. Author: JamieProudfoot.

def LFC_verbose(F,Y):
    """
    Function to calculate LFC (Label-Feature Correlation)
    F :: 'feature function' (N*D matrix)
    Y :: target label (N vector)
    returns lfc :: numerical measure of feature label association
    verbose version
    """
    # Neural Tangent Kernel approximation
    Theta = F@F.T # [N,N]
    Theta -= np.mean(Theta)
    # Label similarity matrix (Kij) = 1 if yi == yj, else -1
    K = 2*(Y.reshape(len(Y),1)==Y).astype(float)-1 # [N,N]
    K -= np.mean(K)
    # Retrun Label-Feature Correlation score
    return np.sum(Theta*K)/(np.linalg.norm(Theta)*np.linalg.norm(K))

#%%

# Succinct version. Author: JamieProudfoot.

def LFC_succinct(F,Y):
    """
    Function to calculate LFC (Label-Feature Correlation)
    F :: 'feature function' (N*D matrix)
    Y :: target label (N vector)
    returns lfc :: numerical measure of feature label association
    succinct version
    """
    Theta=F@F.T
    Theta-=np.mean(Theta)
    K=2*(Y.reshape(len(Y),1)==Y).astype(float)-1
    K-=np.mean(K)
    return np.sum(Theta*K)/(np.linalg.norm(Theta)*np.linalg.norm(K))

#%%

# Testing equivalence of LFC functions

# F = np.random.randn(50,20)
# Y = np.random.randint(2,size=50)

# # print(f"F[:10]:\n{F[:10]}")
# print(f"Y[:10]:\n{Y[:10]}")
# print()

# t0 = datetime.datetime.now()
# print(f"Verbose: {LFC_verbose(F,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))
# print()
# t0 = datetime.datetime.now()
# print(f"Succinct: {LFC_succinct(F,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))
# print()

#%%