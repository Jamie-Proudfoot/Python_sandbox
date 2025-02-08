#%%

import numpy as np
import datetime

#%%

# Original code. 
# Paper: Long-Kai Huang, Junzhou Huang, Yu Rong, Qiang Yang, and Ying Wei.
# Frustratingly easy transferability estimation. 
# In International Conference on Machine Learning, pages 9201–9225. PMLR, 2022

# Comments added here. Author: JamieProudfoot.

def coding_rate(Z, eps=1E-4):
    n, d = Z.shape
    # Compute log det of rate distortion (estimate of entropy)
    (_, rate) = np.linalg.slogdet((np.eye(d) + 1 / (n * eps) * Z.transpose() @ Z))
    return 0.5 * rate

def TrR(Z, y, eps=1E-4):
    # Mean-centered features
    Z = Z - np.mean(Z, axis=0, keepdims=True)
    # Compute marginal rate distortion R(Z;eps)
    RZ = coding_rate(Z, eps)
    RZY = 0.
    K = int(y.max() + 1)
    # !This code assumes a uniform distribution of labels!
    for i in range(K):
        # Compute conditional rate distortion R(Z|y==c;eps)
        RZY += coding_rate(Z[(y == i).flatten()], eps)
    # Compute marginal-conditional rate distortion (esitmate of MI)
    return RZ - RZY / K

#%%

# Verbose version. Author: JamieProudfoot.

def TrR_verbose(F,Y,eps=1e-4):
    """
    Function to calculate TrR (TransRate)
    F :: 'feature function' (N*D matrix)
    Y :: target label (N vector)
    eps :: rate distortion decoding error
    returns trr :: numerical measure of feature label association
    verbose version
    """
    N,Df = F.shape
    F -= np.mean(F,axis=0)
    Ky = int(Y.max()+1)
    # Compute log det of marginal rate distortion R(F;eps)
    RF = (1/2)*np.linalg.slogdet(np.eye(Df)+(1/(N*eps))*(F.T@F))[1]
    Nc = [np.sum(Y==c) for c in range(Ky)]
    # Compute weighted average log det of conditional rate distortions R(F|Y;eps)
    RFY = (1/(2*N))*np.sum([Nc[c]*np.linalg.slogdet(np.eye(Df)+(1/(Nc[c]*eps))*(F[Y==c].T@F[Y==c]))[1] for c in range(Ky)])
    return RF - RFY


#%%

# Succinct version. Author: JamieProudfoot.

def TrR_succinct(F,Y,eps=1e-4):
    """
    Function to calculate TrR (TransRate)
    F :: 'feature function' (N*D matrix)
    Y :: target label (N vector)
    eps :: rate distortion decoding error
    returns trr :: numerical measure of feature label association
    succinct version
    """
    N,Df=F.shape
    F-=np.mean(F,axis=0)
    Ky=int(Y.max()+1)
    RF=(1/2)*np.linalg.slogdet(np.eye(Df)+(1/(N*eps))*(F.T@F))[1]
    Nc=[np.sum(Y==c) for c in range(Ky)]
    RFY=(1/(2*N))*np.sum([Nc[c]*np.linalg.slogdet(np.eye(Df)+(1/(Nc[c]*eps))*(F[Y==c].T@F[Y==c]))[1] for c in range(Ky)])
    return RF - RFY

#%%

# Testing equivalence of LFC functions

# F = np.random.randn(50,20)
# Y = np.random.randint(2,size=50)

# # print(f"F[:10]:\n{F[:10]}")
# print(f"Y[:10]:\n{Y[:10]}")
# print()

# t0 = datetime.datetime.now()
# print(f"Original: {TrR(F,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))
# print()
# t0 = datetime.datetime.now()
# print(f"Verbose: {TrR_verbose(F,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))
# print()
# t0 = datetime.datetime.now()
# print(f"Succinct: {TrR_succinct(F,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))
# print()

#%%