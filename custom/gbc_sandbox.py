#%%

import numpy as np
import datetime

from sklearn.decomposition import PCA

#%%

# Original code. Author: Google-Research. 

# Github: https://github.com/google-research/google-research/blob/b7cb09ba04bf018f1ba1131ccd4c69be617e8e2f/stable_transfer/transferability/gbc.py#L27

# Comments added here. Author: JamieProudfoot.




#%%

# Verbose code. Author: JamieProudfoot.

# Helper function for per-class sample statistics

def class_stats(X, Y, c):
    """
    X :: features (N*D matrix)
    Y :: labels (N matrix)
    Compute sample statistics of all X in a given class c
    returns :: mu_c, var_c
    where mu_c is the sample mean of all X in class c
    and var_c is the sample variance of all X in class c
    """
    Xc = X[Y==c]
    return (np.mean(Xc,axis=0),np.var(Xc,ddof=1,axis=0))

# Helper function for Gaussian Bhattacharyya Coefficient
def BC(mu_cl, var_cl, mu_ck, var_ck):
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
    S = [class_stats(R,Y,c) for c in range(Ky)]
    # Pairs of classes cl != ck without repeats
    triu = np.transpose(np.triu_indices(Ky,1))
    # Array of Bhattacharyya Coefficients
    gbc = [BC(*S[t[0]],*S[t[1]]) for t in triu]
    return -2*sum(gbc)

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
    S = [class_stats(R,Y,c) for c in range(Ky)]
    triu = np.transpose(np.triu_indices(Ky,1))
    gbc = [BC(*S[t[0]],*S[t[1]]) for t in triu]
    return -2*sum(gbc)

#%%

# Testing equivalence of GBC score functions

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