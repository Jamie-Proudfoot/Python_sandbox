#%%
import os
os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np
import datetime

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

#%%

# Verbose code. Author: JamieProudfoot.

def NLEEP_verbose(F,Y,p=0.9,Kv=7):
    """
    F :: 'feature function' (N*Df matrix)
    Y :: Target label (N*r matrix if regression, N vector if classification)
    p :: Dr/Df where Dr is the PCA-reduced dimensionality [0.5, 0.9]
    Kv :: Number of Gaussian components in the GMMs used [1, 7]
    returns :: N-LEEP (Gaussian LEEP) measure to
    predict the success of transfer learning
    verbose version
    """
    # N: number of data points, Df: dimensionality of hidden representation
    N, Df = F.shape
    # Ky: number of target label classes
    Ky = int(np.max(Y) + 1)
    # PCA-reduced dimensions
    Dr = int(np.rint(Df*p))
    # PCA-reduced representation R
    pca = PCA(n_components=Dr)
    R = pca.fit_transform(F)
    # Fit a Gaussian Mixture Model (GMM) to R
    gm = GaussianMixture(n_components=Kv, random_state=0).fit(R)
    # p(v|r) GMM label prediction probability
    A = gm.predict_proba(R)
    # Normalise V so that total sum over V == 1
    normalised_A = A / N
    # Compute joint probability matrix P(y,v)
    joint = (Y == np.vstack(np.arange(Ky)))@normalised_A
    # Compute conditional probability matrix P(y|v) = P(y,v) / P(v)
    conditional = (joint / joint.sum(axis=0)).T
    # Compute EEP (expected empirical prediction)
    marginal = A @ conditional
    EEP = np.array([py[y] for py, y in zip(marginal, Y)])
    # Compute LEEP (Gaussian log of expected empirical prediction)
    leep = np.mean(np.log(EEP))
    return leep

#%%

# Succinct code. Author: JamieProudfoot.

def NLEEP_succinct(F,Y,p=0.9,Kv=7):
    """
    F :: 'feature function' (N*Df matrix)
    Y :: Target label (N*r matrix if regression, N vector if classification)
    p :: Dr/Df where Dr is the PCA-reduced dimensionality [0.5, 0.9]
    Kv :: Number of Gaussian components in the GMMs used [1, 7]
    returns :: N-LEEP (Gaussian LEEP) measure to
    predict the success of transfer learning
    succinct version
    """
    N,Df=F.shape
    Ky=int(np.max(Y)+1)
    Dr=int(np.rint(Df*p))
    pca=PCA(n_components=Dr)
    R=pca.fit_transform(F)
    gm=GaussianMixture(n_components=Kv,random_state=0).fit(R)
    A=gm.predict_proba(R)
    An=A/N
    joint=(Y==np.vstack(np.arange(Ky)))@An
    conditional=(joint/joint.sum(axis=0)).T
    marginal=A@conditional
    EEP=np.array([py[y] for py,y in zip(marginal,Y)])
    leep=np.mean(np.log(EEP))
    return leep

#%%

# Testing equivalence of N-LEEP score functions

# N = 100
# Ky = 2
# Df = 5

# F = np.array(np.random.randint(2,size=(N,Df)),dtype=np.float64)
# Y = np.random.randint(Ky,size=N)

# print(f"F[:10]:\n{F[:10]}")
# print(f"Y[:10]:\n{Y[:10]}")
# print()

# t0 = datetime.datetime.now()
# print(f"Verbose: {NLEEP_verbose(F,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))
# print()
# t0 = datetime.datetime.now()
# print(f"Succinct: {NLEEP_succinct(F,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))

#%%