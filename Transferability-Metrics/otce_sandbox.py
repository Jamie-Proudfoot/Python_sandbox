#%%
import numpy as np
import datetime
import geomloss
import torch
import math

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import ot

from scipy.stats import entropy
from sklearn.metrics.pairwise import euclidean_distances

#%%

# Original code. Author: tanyang1231. Github: https://github.com/tanyang1231/OTCE_Transferability_CVPR21/

# Comments added here. Author: JamieProudfoot.

def compute_coupling(X_src, X_tar, Y_src, Y_tar):
    # Compute cost function
    cost_function = lambda x, y: geomloss.utils.squared_distances(x, y)
    C = cost_function(X_src,X_tar)
    # Compute optimal transportation (coupling) matrix
    P = ot.emd(ot.unif(np.array(X_src).shape[0]), ot.unif(np.array(X_tar).shape[0]), np.array(C), numItermax=100000)
    # Compute 1-Wasserstein distance
    W = np.sum(P*np.array(C.numpy()))
    return P,W


def compute_CE(P, Y_src, Y_tar):
    # Set of possible Y_src values
    src_label_set = set(sorted(list(Y_src.flatten())))
    # Set of possible Y_tar values
    tar_label_set = set(sorted(list(Y_tar.flatten())))

    # joint distribution of source and target label
    P_src_tar = np.zeros((np.max(Y_src)+1,np.max(Y_tar)+1))

    for y1 in src_label_set:
        y1_idx = np.where(Y_src==y1)
        for y2 in tar_label_set:
            y2_idx = np.where(Y_tar==y2)

            RR = y1_idx[0].repeat(y2_idx[0].shape[0])
            CC = np.tile(y2_idx[0], y1_idx[0].shape[0])

            P_src_tar[y1,y2] = np.sum(P[RR,CC])

    # marginal distribution of source label
    P_src = np.sum(P_src_tar,axis=1)

    ce = 0.0
    for y1 in src_label_set:
        P_y1 = P_src[y1]
        for y2 in tar_label_set:
            # Avoid log(0)
            if P_src_tar[y1,y2] != 0:
                # Manual entropy calculation
                ce += -(P_src_tar[y1,y2] * math.log(P_src_tar[y1,y2] / P_y1))
    return ce

def OTCE(Fs,F,Z,Y,lamb=(1,1,1)):
    """
    Fs :: source hidden representations (N*Df matrix)
    F :: target hidden representations (Ft) (N*Df matrix)
    Z :: source output labels (Ys) (N vector)
    Y :: target output labels (Yt) (N vector)
    lamb :: tuple of hyperparameters (b,lambda1,lambda2)
    returns otce :: numerical measure of source dataset
    to target dataset transferability
    original version
    Author: tanyang1231. 
    Github: https://github.com/tanyang1231/OTCE_Transferability_CVPR21/
    """
    Fs = torch.from_numpy(Fs)
    F = torch.from_numpy(F)
    P, WD = compute_coupling(Fs,F,Z,Y)
    print(f"Wasserstein distance: {WD}")
    CE = compute_CE(P,Z,Y)
    print(f"Conditional entropy: {CE}")
    return np.sum(np.array(lamb)*np.array((1,WD,CE)))

#%%

# Verbose code. Author: JamieProudfoot.

def OTCE_verbose(Fs,F,Z,Y,lamb=(1,1,1)):
    """
    Fs :: source hidden representations (N*Df matrix)
    F :: target hidden representations (Ft) (N*Df matrix)
    Z :: source output labels (Ys) (N vector)
    Y :: target output labels (Yt) (N vector)
    lamb :: tuple of hyperparameters (b,lambda1,lambda2)
    returns otce :: numerical measure of source dataset
    to target dataset transferability
    verbose version
    """
    # Cost matrix
    C = np.square(euclidean_distances(Fs,F))
    # Sinkhorn algorithm 1-Wasserstein distance
    WD = ot.emd2(ot.unif(len(Fs)),ot.unif(len(F)),C)
    print(f"Wasserstein distance: {WD}")
    # Compute conditional entropy (CE)
    YZ = np.column_stack((Y,Z))
    CE = entropy(np.unique(YZ,return_counts=True,axis=0)[1]/len(YZ)) \
    - entropy(np.unique(Z,return_counts=True,axis=0)[1]/len(Z))
    print(f"Conditional entropy: {CE}")
    # Return OTCE (Optimal Transport-Based Conditional Entropy)
    return np.sum(np.array(lamb)*np.array((1,WD,CE)))


#%%

# Succinct code. Author: JamieProudfoot.

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
    C = np.square(euclidean_distances(Fs,F))
    WD = ot.emd2(ot.unif(len(Fs)),ot.unif(len(F)),C)
    print(f"Wasserstein distance: {WD}")
    YZ=np.column_stack((Y,Z))
    CE=entropy(np.unique(YZ,return_counts=True,axis=0)[1]/len(YZ)) \
    -entropy(np.unique(Z,return_counts=True,axis=0)[1]/len(Z))
    print(f"Conditional entropy: {CE}")
    return np.sum(np.array(lamb)*np.array((1,WD,CE)))

#%%

# Testing equivalence of OTCE functions

# Fs = np.random.randn(50,2)
# F = np.random.randn(50,2)
# Z = np.random.randint(2,size=50)
# Y = np.random.randint(2,size=50)

# print(f"Fs[:10]:\n{Fs[:10]}")
# print(f"Z[:10]:\n{Z[:10]}")
# print(f"F[:10]:\n{F[:10]}")
# print(f"Y[:10]:\n{Y[:10]}")
# print()

# t0 = datetime.datetime.now()
# print(f"Original: {OTCE(Fs,F,Z,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))
# print()
# t0 = datetime.datetime.now()
# print(f"Verbose: {OTCE_verbose(Fs,F,Z,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))
# print()
# t0 = datetime.datetime.now()
# print(f"Succinct: {OTCE_succinct(Fs,F,Z,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))

#%%