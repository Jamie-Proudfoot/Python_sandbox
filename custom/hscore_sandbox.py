#%%

import numpy as np
import datetime
from sklearn.covariance import LedoitWolf

#%%

# Original code. Author: YaojieBao. Github: https://github.com/YaojieBao/An-Information-theoretic-Metric-of-Transferability

# Comments added here. Author: JamieProudfoot.

# Helper function for covariance
def getCov(X):
    # Specify dtype to avoid instability in pinv calculation
    X=np.array(X,dtype=np.float64)
    X_mean=X-np.mean(X,axis=0,keepdims=True)
    cov = np.divide(np.dot(X_mean.T, X_mean), len(X)-1) 
    return cov
# Can replace this with np.cov(X.T) or np.cov(X,rowvar=False)

# H-score function
def HScore(F,Y):
    # Overall covariance for F
    Covf=getCov(F)
    # Covf=np.cov(F.T) # using numpy
    # List of class labels
    alphabetY=list(set(Y))
    # Initialise g (replace with g=np.zeros_like(F,dtype=np.float64)) because default data type is int or dtype=int
    # Can replace with g=np.zeros(F.shape())
    g=np.zeros_like(F)
    for y in alphabetY:
        Ef_y=np.mean(F[Y==y, :], axis=0)
        g[Y==y]=Ef_y
    # Assuming F is a D*N matrix (features by data). Rows = features, Columns = data points
    # Inter-class covariance for F
    Covg=getCov(g)
    # Covg=np.cov(g.T) # using numpy
    # H-score as defined by the equation in the paper (Definition 2, Bao, Yaojie, et al. (2019).)
    score=np.trace(np.dot(np.linalg.pinv(Covf,rcond=1e-15), Covg)) # rcond=1e-15 is the default value in pinv
    return score

#%%

# Modified code (verbose). Author: JamieProudfoot.

def HScore_verbose(F,Y): 
    """
    Function to calculate H-score
    F :: 'feature function' (N*D matrix)
    Y :: target label (N vector)
    returns hscore :: numerical measure of feature label association
    verbose version
    """
    # Overall covariance for F
    Covf = np.cov(F,rowvar=False)
    # List of class labels
    alphabetY=list(set(Y))
    # Average of F over K classes
    g=np.zeros_like(F,dtype=np.float64)
    for y in alphabetY: 
        class_avg=np.mean(F[Y==y, :], axis=0)
        g[Y==y]=class_avg
    # Inter-class covariance for F
    Covg = np.cov(g,rowvar=False)
    # H-score as defined by the equation in the paper (Definition 2, Bao, Yaojie, et al. (2019).)
    hscore = np.trace(np.linalg.pinv(Covf)@Covg)
    return hscore

 #%%

# Modified code (succinct). Author: JamieProudfoot.

def HScore_succinct(F,Y):
    """
    Function to calculate H-score
    F :: 'feature function' (N*D matrix)
    Y :: target label (N vector)
    returns hscore :: numerical measure of feature label association
    succinct version
    """
    g=np.zeros_like(F,dtype=np.float64)
    for y in set(Y): g[Y==y]=np.mean(F[Y==y],axis=0)
    return np.trace(np.linalg.pinv(np.cov(F,rowvar=False))@np.cov(g,rowvar=False))

#%%

# Regularised H-Score calculation 
# Following the suggestions in this paper: https://arxiv.org/abs/2110.06893
# Uses LeDoit Wolf algorithm to compute a more stable correlation matrix

def HScore_regularised(F,Y):
    """
    Function to calculate H-score according to https://arxiv.org/abs/2110.06893
    F :: 'feature function' (N*D matrix)
    Y :: target label (N vector)
    returns hscore :: numerical measure of feature label association
    """
    F=F.astype(np.float64)
    g=np.zeros_like(F,dtype=np.float64)
    cov=LedoitWolf().fit(F-np.mean(F,axis=0,keepdims=True))
    for y in set(Y): g[Y==y]=np.mean(F[Y==y],axis=0)
    return np.trace(np.linalg.pinv(cov.covariance_)@((1-cov.shrinkage_)*np.cov(g,rowvar=False)))
#%%

# Testing equivalence of H-score functions

# F = np.array([[1,0,1,1,1],[0,1,1,0,1],[1,1,1,1,1],[0,0,0,1,1]],dtype=np.float64)
# Y = np.array([1,1,0,0])

# print(f"F:\n{F}")
# print(f"Y:\n{Y}")
# print()

# t0 = datetime.datetime.now()
# print(f"Original: {HScore(F,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))
# print()
# t0 = datetime.datetime.now()
# print(f"Verbose: {HScore_verbose(F,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))
# print()
# t0 = datetime.datetime.now()
# print(f"Succinct: {HScore_succinct(F,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))
# print()
# t0 = datetime.datetime.now()
# print(f"Regularised: {HScore_regularised(F,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))
# print()

# %%