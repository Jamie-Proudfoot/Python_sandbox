#%%
import numpy as np
import datetime
import ot

from scipy.stats import entropy

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

#%%

def OTCE_verbose(Fs,F,Z,Y,lamb=(1,1,1)):
    """
    feature_extractor :: source feature function (fs)
    Xs :: source input data (N*Ds matrix)
    X :: target input data (Xt) (N*Dt matrix)
    Z :: source output labels (Ys) (N vector)
    Y :: target output labels (Yt) (N vector)
    lamb :: tuple of hyperparameters (b,lambda1,lambda2)
    returns otce :: numerical measure of source dataset
    to target dataset transferability
    verbose version
    """
    # Cost matrix
    C = ot.dist(Fs,F)
    # Sinkhorn algorithm 1-Wasserstein distance
    WD = ot.emd2(ot.unif(len(Fs)),ot.unif(len(F)),C)
    # Compute conditional entropy (CE)
    YZ = np.column_stack((Y,Z))
    CE = entropy(np.unique(YZ,return_counts=True,axis=0)[1]/len(YZ)) \
    - entropy(np.unique(Z,return_counts=True,axis=0)[1]/len(Z))
    # Return OTCE (Optimal Transport-Based Conditional Entropy)
    return np.sum(np.array(lamb)*np.array((1,WD,CE)))


#%%

def OTCE_succinct(Fs,F,Z,Y,lamb=(1,1,1)):
    """
    Fs :: source hidden representations (N*Df matrix)
    F :: target hidden representations (Ft) (N*Df matrix)
    Z :: source output labels (Ys) (N vector)
    Y :: target output labels (Yt) (N vector)
    lamb :: tuple of hyperparameters (b,lambda1,lambda2)
    returns otce :: numerical measure of source dataset
    to target dataset transferability
    succinct version
    """
    C = ot.dist(Fs,F)
    WD = ot.emd2(ot.unif(len(Fs)),ot.unif(len(F)),C)
    YZ=np.column_stack((Y,Z))
    CE=entropy(np.unique(YZ,return_counts=True,axis=0)[1]/len(YZ)) \
    -entropy(np.unique(Z,return_counts=True,axis=0)[1]/len(Z))
    return np.sum(np.array(lamb)*np.array((1,WD,CE)))

#%%

# Testing equivalence of OTCE functions

Fs = np.random.randn(50,1)
F = np.random.randn(50,1)
Z = np.random.randint(2,size=50)
Y = np.random.randint(2,size=50)

print(f"Fs[:10]:\n{Fs[:10]}")
print(f"Z[:10]:\n{Z[:10]}")
print(f"F[:10]:\n{F[:10]}")
print(f"Y[:10]:\n{Y[:10]}")
print()

t0 = datetime.datetime.now()
print(f"Verbose: {OTCE_verbose(Fs,F,Z,Y)}")
t1 = datetime.datetime.now()
print((t1-t0))
print()
t0 = datetime.datetime.now()
print(f"Succinct: {OTCE_succinct(Fs,F,Z,Y)}")
t1 = datetime.datetime.now()
print((t1-t0))

#%%