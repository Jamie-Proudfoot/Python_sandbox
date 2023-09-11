#%%

import numpy as np
import datetime

from sklearn.decomposition import PCA
from scipy.special import softmax
from scipy.optimize import minimize

import matplotlib.pyplot as plt

#%%

# Verbose code. Author: JamieProudfoot.

# Function to compute the regularised empirical risk
# Categorical cross entropy loss with L2 regularisation
def RER(W,R,Yb,L,v0):
    """
	W :: linear parameters, [(Ky+1)*Dr] vector
         W = [w,b] where w are weights and b are biases
	R :: features, (N*Dr) matrix
	Yb :: one-hot encoded target training labels, (N*Ky) matrix
	L :: lambda parameter
	v0 :: variance estimate parameter
	returns rer :: regularised empirical risk
	"""
    N,Dr = R.shape
    Ky = Yb.shape[1]
	# w weights shape (Dr,Ky)
    w = W[:(Dr*Ky)].reshape(Dr,Ky)
	# b bias shape (Ky)
    b = W[(Dr*Ky):]
	# NN output linear approximation
    G = R@w+b
	# Categorial crossentropy loss function
    loss = (1/N)*np.sum(-np.sum(Yb*G,axis=1)+np.log(np.sum(np.exp(G),axis=1)))
	# L2 regularisation term
    l2 = np.sum(np.square(W))
    return loss + (1/(2*L*v0))*l2

# Function to comput the gradient of regularised empirical risk
# Categorical cross entropy loss with L2 regularisation
def dRER(W,R,Yb,L,v0):
    """
	W :: linear parameters, (Ky+1)*Dr vector
	R :: features, (N*Dr) matrix
	Yb :: one-hot encoded target training labels, (N*Ky) matrix
	L :: lambda parameter
	v0 :: variance estimate parameter
	returns drer :: gradient of regularised empirical risk
	"""
    N,Dr = R.shape
    Ky = Yb.shape[1]
	# w weights shape (Dr,Ky)
    w = W[:(Dr*Ky)].reshape(Dr,Ky)
	# b bias shape (Ky)
    b = W[(Dr*Ky):]
	# NN output linear approximation
    G = R@w+b
    # Compute label probabilities
    A = softmax(G,axis=1)
    # Gradient of crossentropy wrt weights
    dLdw = (1/N)*R.T@(A-Yb) + (1/(L*v0))*w
    # Gradient of crossentropy wrt biases
    dLdb = (1/N)*np.sum(A-Yb,axis=0) + (1/(L*v0))*b
    return np.concatenate((dLdw.flatten(),dLdb))

def clbk(W):
    # Custom processing and display of iteration values
    clbk.iterations.append(clbk.iteration)
    clbk.weights.append(W)
    clbk.iteration += 1

# Function to compute PACTran_Gaussian (with fixed hyperparameters L, V0)
def PACTran_verbose(F,Y,p=0.9):
    """
	F :: hidden-embedding features, (N*Df) matrix
	Y :: target training labels, N vector
	"""
    # N : number of points, Df : feature dimension
    N,Df = F.shape
    # Dr: PCA-reduced feature dimension
    Dr = int(np.rint(Df*p))
	# L: lambda parameter
    L = N*Dr/20
	# v0: estimated variance parameter
    v0 = 100/Dr
	# Ky : number of classes
    Ky = int(Y.max()+1)
	# One-hot encoded labels
    Yb = (np.arange(Ky)==Y.reshape(N,1)).astype(int)
	# Generate PCA-reduced features
    pca = PCA(n_components=Dr)
    R = pca.fit_transform(F)
	# L-BFGS optimisation of categorical crossentropy loss
    # Random initialisation
    W0 = np.random.normal(size=((Dr+1)*Ky))
    clbk.iteration = 0
    clbk.iterations = []
    clbk.weights = []
    opt = minimize(RER,W0,args=(R,Yb,L,v0),
                    method='L-BFGS-B',jac=dRER,callback=clbk)
    print(opt)
    iterations = np.array(clbk.iterations)
    weights = np.array(clbk.weights)
    RERs = np.array([RER(w,R,Yb,L,v0) for w in weights])
    plt.plot(iterations,RERs)
    plt.savefig
	# Optimised parameters
    Wopt = opt.x
    wopt = Wopt[:(Dr*Ky)].reshape(Dr,Ky)
    bopt = Wopt[(Dr*Ky):]
    # Optimised loss
    RERopt = opt.fun
	# Optimised label predictions
    Gopt = R@wopt+bopt
    Aopt = softmax(Gopt,axis=1)
	# Trace of optimised RER Hessian
    d2Ldw2 = (1/N)*np.square(R).T@(Aopt-np.square(Aopt))
    d2Ldb2 = (1/N)*np.sum(Aopt-np.square(Aopt),axis=0)
    TrHess = np.sum(d2Ldw2) + np.sum(d2Ldb2)
	# Return PACTran_Gauss result
    return RERopt + (Ky*Dr/(2*L))*np.log(1+(L*v0/(Ky*Dr))*TrHess)

#%%

# Testing equivalence of PACTran score functions

N = 100
Ky = 4
Df = 6

F = np.array(np.random.randint(2,size=(N,Df)),dtype=np.float64)
Y = np.random.randint(Ky,size=N)

print(f"F[:10]:\n{F[:10]}")
print(f"Y[:10]:\n{Y[:10]}")
print()

t0 = datetime.datetime.now()
print(f"Verbose: {PACTran_verbose(F,Y)}")
t1 = datetime.datetime.now()
print((t1-t0))

#%%
