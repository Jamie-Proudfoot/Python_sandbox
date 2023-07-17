#%%

import numpy as np
import datetime

#%%

# Original code. Author: YaojieBao. Github: https://github.com/YaojieBao/An-Information-theoretic-Metric-of-Transferability

# Comments added here. Author: JamieProudfoot.

# Helper function for covariance
def getCov(X):
    X_mean=X-np.mean(X,axis=0,keepdims=True)
    cov = np.divide(np.dot(X_mean.T, X_mean), len(X)-1) 
    return cov
# Can replace this with np.cov(X.T)

# H-score function
def HScore(f,Y):
    # Overall covariance for f
    Covf=getCov(f)
    # Covf=np.cov(f.T) # using numpy
    # List of class labels
    alphabetY=list(set(Y))
    # Initialise g (replace with g=np.zeros_like(f,dtype=np.float32)) because default data type is int or dtype=int
    # Can replace with g=np.zeros(f.shape())
    g=np.zeros_like(f,dtype=np.float32)
    print(g)
    for y in alphabetY:
        Ef_y=np.mean(f[Y==y, :], axis=0)
        g[Y==y]=Ef_y
    # Assuming f is a n*d matrix (data by features). Rows = data points, Columns = features
    # Inter-class covariance for f
    Covg=getCov(g)
    # Covg=np.cov(g.T) # using numpy
    # H-score as defined by the equation in the paper (Definition 2, Bao, Yaojie, et al. (2019).)
    score=np.trace(np.dot(np.linalg.pinv(Covf,rcond=1e-15), Covg)) # rcond=1e-15 is the default value in pinv
    return score

#%%

# Modified code (verbose). Author: JamieProudfoot.

def HScore_verbose(f,Y): 
    """
    Function to calculate H-score
    f :: 'feature function' (n*d matrix)
    Y :: target label (n vector)
    returns hscore :: numerical measure of feature label association
    verbose version
    """
    # Overall covariance for f
    Covf = np.cov(f.T)
    # List of class labels
    alphabetY=list(set(Y))
    # Average of f over k classes
    g=np.zeros_like(f,dtype=np.float32)
    for y in alphabetY: 
        class_avg=np.mean(f[Y==y, :], axis=0)
        g[Y==y]=class_avg
    # Inter-class covariance for f
    Covg = np.cov(g.T)
    # H-score as defined by the equation in the paper (Definition 2, Bao, Yaojie, et al. (2019).)
    hscore = np.trace(np.linalg.pinv(Covf)@Covg)
    return hscore

 #%%

# Modified code (succinct). Author: JamieProudfoot.

def HScore_succinct(f,Y):
    """
    Function to calculate H-score
    f :: 'feature function' (n*d matrix)
    Y :: target label (n vector)
    returns hscore :: numerical measure of feature label association
    succinct version
    """
    g=np.zeros_like(f,dtype=np.float32)
    for y in set(Y): g[Y==y]=np.mean(f[Y==y],axis=0)
    return np.trace(np.linalg.pinv(np.cov(f.T))@np.cov(g.T))

#%%

# Testing equivalence of H-score functions

# f = np.array([[1,0,1,1],[0,1,1,0],[1,1,1,1],[0,0,0,1]])
# Y = np.array([1,1,0,0])

# print(f"f:\n{f}")
# print(f"Y:\n{Y}")
# print()

# t0 = datetime.datetime.now()
# print(f"Original: {HScore(f,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))
# print()
# t0 = datetime.datetime.now()
# print(f"Verbose: {HScore_verbose(f,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))
# print()
# t0 = datetime.datetime.now()
# print(f"Succinct: {HScore_succinct(f,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))

# %%