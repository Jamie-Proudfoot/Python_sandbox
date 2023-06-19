#%%

import numpy as np

#%%

# Original code. Author: YaojieBao. Github: https://github.com/YaojieBao/An-Information-theoretic-Metric-of-Transferability

# Comments added here. Author: JamieProudfoot.

# Helper function for covariance
def getCov(X):
    X_mean=X-np.mean(X,axis=0,keepdims=True)
    cov = np.divide(np.dot(X_mean.T, X_mean), len(X)-1) 
    return cov
# Can just replace this with np.cov(X.T)

# H-score function
def getHscore(f,Z):
    # Overall covariance for f
    Covf=getCov(f)
    # Covf=np.cov(f.T) # using numpy
    # List of class labels
    alphabetZ=list(set(Z))
    # Initialise g (replace with g=np.zeros_like(f,dtype=np.float64)) because default data type is int or dtype=int
    # Can replace with g=np.zeros(f.shape())
    g=np.zeros_like(f)
    for z in alphabetZ:
        Ef_z=np.mean(f[Z==z, :], axis=0)
        g[Z==z]=Ef_z
    # Assuming f is a n*d matrix (data by features). Rows = data points, Columns = features
    # Inter-class covariance for f
    Covg=getCov(g)
    # Covg=np.cov(g.T) # using numpy
    # H-score as defined by the equation in the paper (Definition 2, Bao, Yaojie, et al. (2019).)
    score=np.trace(np.dot(np.linalg.pinv(Covf,rcond=1e-15), Covg)) # rcond=1e-15 is the default value in pinv
    return score

#%%

# Modified code (verbose). Author: JamieProudfoot.

def getHscore_verbose(f,Z): 
    """
    Function to calculate H-score
    f :: 'feature function' (n*d matrix)
    Z :: class labels (k vector)
    returns hscore :: numerical measure of feature label association
    verbose version
    """
    # Overall covariance for f
    Covf = np.cov(f.T)
    # List of class labels
    alphabetZ=list(set(Z))
    # Average of f over k classes
    g=np.zeros_like(f,dtype=np.float64)
    for z in alphabetZ: g[Z==z]=np.mean(f[Z==z],axis=0)
    # Inter-class covariance for f
    Covg = np.cov(g.T)
    # H-score as defined by the equation in the paper (Definition 2, Bao, Yaojie, et al. (2019).)
    hscore = np.trace(np.dot(np.linalg.pinv(Covf), Covg))
    return hscore

 #%%

# Modified code (succinct). Author: JamieProudfoot.

def getHscore_succinct(f,Z):
    """
    Function to calculate H-score
    f :: feature function (n*d matrix)
    Z :: class labels (k vector)
    returns hscore :: numerical measure of feature label association
    succinct version
    """
    g=np.zeros_like(f,dtype=np.float64)
    for z in set(Z): g[Z==z]=np.mean(f[Z==z],axis=0)
    return np.trace(np.dot(np.linalg.pinv(np.cov(f.T)),np.cov(g.T)))

#%%

# Testing equivalence of H-score functions

f = np.array([[1,0,1,1],[0,1,1,0],[1,1,1,1],[0,0,0,1]])
Z = np.array([1,1,0,0])

print(f)
print(Z)

print(f"Original: {getHscore(f,Z)}")
print(f"Verbose: {getHscore_verbose(f,Z)}")
print(f"Succinct: {getHscore_verbose(f,Z)}")
# %%
