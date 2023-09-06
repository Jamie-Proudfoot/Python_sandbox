#%%

import numpy as np
import datetime

from sklearn.decomposition import PCA

#%%

# Verbose code. Author: JamieProudfoot.

def class_stats(X, Y):
    """
    X :: features (N*D matrix)
    Y :: labels (N matrix)
    Compute sample statistics of all X in a given class c
    returns :: mu_c, var_c
    where mu_c is the sample mean of all X in class c
    and var_c is the sample variance of all X in class c
    """
    Ky = int(Y.max() + 1)
    means = [np.mean(X[Y==c],axis=0) for c in range(Ky)]
    vars = [np.var(X[Y==c],ddof=1,axis=0) for c in range(Ky)]
    return means, vars

def BC(mu_cl, mu_ck, var_cl, var_ck):
    """
    X :: features (N*D matrix)
    Y :: labels (N matrix)
    Compute Gaussian Bhattacharyya Coefficient
    between two classes cl and ck
    returns :: gbc_lk (Gaussian Bhattacharyya Coefficient between cl and ck)
    """
    # Averaged class variance
    var = 0.5*(var_cl+var_ck)
    # Bhattacharyya Distance
    DB_lk = np.sum(0.125*(mu_cl-mu_ck)**2/var \
        + 0.5*np.log(var) \
        - 0.25*np.log(var_ck) \
        - 0.25*np.log(var_cl))
    return np.exp(-DB_lk)

def GBC_verbose(F, Y, p=0.9):
    """
    F :: 'feature function' (N*Df matrix)
    Y :: Target label (N*r matrix if regression, N vector if classification)
    p :: Dr/Df where Dr is the PCA-reduced dimensionality [0.5, 0.9]
    returns :: GBC (Gaussian Bhattacharyya Coefficient) measure to
    predict the success of transfer learning
    verbose version
    """
    # N: number of data points, Df: dimensionality of hidden representation
    N, Df = F.shape
    # Ky: number of target label classes
    Ky = int(Y.max() + 1)
    # PCA-reduced dimensions
    Dr = int(np.rint(Df*p))
    # PCA-reduced representation R
    pca = PCA(n_components=Dr)
    R = pca.fit_transform(F)
    # Compute class statistics
    means, vars = class_stats(R,Y)
    # Pairs of classes cl != ck without repeats
    triu = np.transpose(np.triu_indices(Ky,1))
    # Array of Bhattacharyya Coefficients
    b = [BC(means[t[0]],means[t[1]],vars[t[0]],vars[t[1]]) for t in triu]
    return -2*sum(b)

#%%

# Succinct code. Author: JamieProudfoot.

def GBC_succinct(F,Y,p=0.9):
    """
    F :: 'feature function' (N*Df matrix)
    Y :: Target label (N*r matrix if regression, N vector if classification)
    p :: Dr/Df where Dr is the PCA-reduced dimensionality [0.5, 0.9]
    returns :: GBC (Gaussian Bhattacharyya Coefficient) measure to
    predict the success of transfer learning
    verbose version
    """
    N,Df=F.shape
    Ky=int(Y.max()+1)
    Dr=int(np.rint(Df*p))
    pca=PCA(n_components=Dr)
    R=pca.fit_transform(F)
    means, vars = class_stats(R,Y)
    triu = np.transpose(np.triu_indices(Ky,1))
    b = [BC(means[t[0]],means[t[1]],vars[t[0]],vars[t[1]]) for t in triu]
    return -2*sum(b)

#%%

# Testing equivalence of N-LEEP score functions

N = 100
Ky = 4
Df = 5

F = np.array(np.random.randint(2,size=(N,Df)),dtype=np.float64)
Y = np.random.randint(Ky,size=N)

print(f"F[:10]:\n{F[:10]}")
print(f"Y[:10]:\n{Y[:10]}")
print()

t0 = datetime.datetime.now()
print(f"Verbose: {GBC_verbose(F,Y)}")
t1 = datetime.datetime.now()
print((t1-t0))
print()
t0 = datetime.datetime.now()
print(f"Succinct: {GBC_succinct(F,Y)}")
t1 = datetime.datetime.now()
print((t1-t0))

#%%