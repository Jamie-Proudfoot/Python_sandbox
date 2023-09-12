#%%

import numpy as np
import time
import datetime

from sklearn.decomposition import PCA
from scipy.special import softmax
from scipy.special import logsumexp
from scipy.optimize import minimize

import matplotlib.pyplot as plt


#%%

# Original code. Author: Google-Research. 

# Github: https://github.com/google-research/pactran_metrics

# Comments added here. Author: JamieProudfoot.

def one_hot(a):
  b = np.zeros((a.size, a.max()+1))
  b[np.arange(a.size), a] = 1.
  return b

def PACTran(F,Y,lda_factor,p=0.9):
  """Compute the PAC_Gauss score with diagonal variance."""
  starttime = time.time()
  Ky = Y.max()+1
  Yb = one_hot(Y)  # [N, v]

  # PCA reduction of F
  Dr = int(np.rint(F.shape[-1]*p))
  pca=PCA(n_components=Dr)
  F=pca.fit_transform(F)
  
  meanF = np.mean(F, axis=0, keepdims=True)
  F -= meanF  # [N,Ky]

  N = F.shape[0]
  Df = F.shape[-1]
  KD = Df * Ky
  ldas2 = lda_factor * N
  dinv = 1. / float(Df)

  # optimizing log lik + log prior
  def pac_loss_fn(W):
    W = np.reshape(W, [Df + 1, Ky])

    w = W[:Df, :]
    b = W[Df:, :]
    logits = np.matmul(F, w) + b

    log_qz = logits - logsumexp(logits, axis=-1, keepdims=True)
    xent = np.sum(np.sum(
        Yb * (np.log(Yb + 1e-10) - log_qz), axis=-1)) / N
    loss = xent + 0.5 * np.sum(np.square(w)) / ldas2
    return loss

  # gradient of xent + l2
  def pac_grad_fn(W):
    W = np.reshape(W, [Df + 1, Ky])

    w = W[:Df, :]
    b = W[Df:, :]
    logits = np.matmul(F, w) + b

    grad_f = softmax(logits, axis=-1)  # [N, Ky]
    grad_f -= Yb
    grad_f /= N
    grad_w = np.matmul(F.transpose(), grad_f)  # [D, Ky]
    grad_w += w / ldas2

    grad_b = np.sum(grad_f, axis=0, keepdims=True)  # [1, Ky]
    grad = np.ravel(np.concatenate([grad_w, grad_b], axis=0))
    return grad

  # 2nd gradient of W (elementwise)
  def pac_grad2(W):
    W = np.reshape(W, [Df + 1, Ky])

    w = W[:Df, :]
    b = W[Df:, :]
    logits = np.matmul(F, w) + b

    prob_logits = softmax(logits, axis=-1)  # [N, Ky]
    grad2_f = prob_logits - np.square(prob_logits)  # [N, Ky]
    xx = np.square(F)  # [N, D]

    grad2_w = np.matmul(xx.transpose(), grad2_f)  # [D, Ky]
    grad2_w += 1. / ldas2
    grad2_b = np.sum(grad2_f, axis=0, keepdims=True)  # [1, Ky]
    grad2 = np.ravel(np.concatenate([grad2_w, grad2_b], axis=0))
    return grad2

  kernel_shape = [Df, Ky]
  W = np.random.normal(size=kernel_shape) * 0.03
  W_1d = np.ravel(np.concatenate(
      [W, np.zeros([1, Ky])], axis=0))

  W_1d = minimize(
      pac_loss_fn, W_1d, method="L-BFGS-B",
      jac=pac_grad_fn,
      options=dict(maxiter=100), tol=1e-6).x
  
  pac_opt = pac_loss_fn(W_1d)
  endtime_opt = time.time()

  h = pac_grad2(W_1d)
  sigma2_inv = np.sum(h) * ldas2  / KD + 1e-10
  endtime = time.time()

  if lda_factor == 10.:
    s2s = [1000., 100.]
  elif lda_factor == 1.:
    s2s = [100., 10.]
  elif lda_factor == 0.1:
    s2s = [10., 1.]
    
  returnv = []
  for s2_factor in s2s:
    s2 = s2_factor * dinv
    pac_gauss = pac_opt + 0.5 * KD / ldas2 * s2 * np.log(
        sigma2_inv)
    
    # the first item is the pac_gauss metric
    # the second item is the linear metric (without trH)
    returnv += [("pac_gauss_%.1f" % lda_factor, pac_gauss),
                ("time", endtime - starttime),
                ("pac_opt_%.1f" % lda_factor, pac_opt),
                ("time", endtime_opt - starttime)]
  return returnv, W_1d

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
    """
    Custom callback function for storing intermediate
    L-BFGS optimisation steps
    """
    clbk.iterations.append(clbk.iteration)
    clbk.weights.append(W)
    clbk.iteration += 1

# Function to compute PACTran_Gaussian (with fixed hyperparameters L, v0)
def PACTran_verbose(F,Y,p=0.9):
    """
	F :: hidden-embedding features, (N*Df) matrix
	Y :: target training labels, N vector
    p :: Dr/Df where Dr is the PCA-reduced dimensionality [0.5, 0.9]
    returns :: PACTran_Gaussian (PAC-Transferability measure with 
    Gaussain prior) to predict the success of transfer learning
    verbose version
	"""
    # N : number of points, Df : feature dimension
    N,Df = F.shape
    # Dr: PCA-reduced feature dimension
    Dr = int(np.rint(Df*p))
    # L: lambda parameter
    L = N*Dr/10
    # v0: estimated variance parameter
    v0 = 100/Dr
    # Ky : number of classes
    Ky = int(Y.max()+1)
    # One-hot encoded labels
    Yb = (np.arange(Ky)==Y.reshape(N,1)).astype(int)
    # Generate PCA-reduced features
    pca = PCA(n_components=Dr)
    R = pca.fit_transform(F)
    R -= np.mean(R,axis=0)
    # L-BFGS optimisation of categorical crossentropy loss
    # Random initialisation
    w0 = np.random.normal(size=(Dr*Ky))*0.03
    b0 = np.zeros(Ky)
    W0 = np.concatenate((w0.flatten(),b0))
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
    # Optimised parameters
    Wopt = opt.x
    wopt = Wopt[:(Dr*Ky)].reshape(Dr,Ky)
    bopt = Wopt[(Dr*Ky):]
    # Optimised loss
    RERopt = opt.fun
    # Optimised label predictions
    Gopt = R@wopt+bopt
    Aopt = softmax(Gopt,axis=1)
    # Hessian components (original paper omits division by N)
    d2Ldw2 = (1/N)*np.square(R).T@(Aopt-np.square(Aopt))
    d2Ldb2 = (1/N)*np.sum(Aopt-np.square(Aopt),axis=0)
    # Trace of optimised RER Hessian
    TrHess = np.sum(d2Ldw2) + np.sum(d2Ldb2)
    # Return PACTran_Gauss result
    return RERopt + (Ky*Dr/(2*L))*np.log(1+(L*v0/(Ky*Dr))*TrHess)

#%%

# Conise code. Author: JamieProudfoot.

# Function to compute PACTran_Gaussian (with fixed hyperparameters L, v0)
def PACTran_concise(F,Y,p=0.9):
    """
	F :: hidden-embedding features, (N*Df) matrix
	Y :: target training labels, N vector
    p :: Dr/Df where Dr is the PCA-reduced dimensionality [0.5, 0.9]
    returns :: PACTran_Gaussian (PAC-Transferability measure with 
    Gaussain prior) to predict the success of transfer learning
    concise version
	"""
    N,Df=F.shape
    Dr=int(np.rint(Df*p))
    L=N*Dr/10
    v0=100/Dr
    Ky=int(Y.max()+1)
    Yb=(np.arange(Ky)==Y.reshape(N,1)).astype(int)
    pca=PCA(n_components=Dr)
    R=pca.fit_transform(F)
    R-=np.mean(R,axis=0)
    W0 = np.concatenate((0.03*np.random.normal(size=(Dr*Ky)),np.zeros(Ky)))
    opt=minimize(RER,W0,args=(R,Yb,L,v0),method='L-BFGS-B',jac=dRER)
    Wopt=opt.x
    wopt=Wopt[:(Dr*Ky)].reshape(Dr,Ky)
    bopt=Wopt[(Dr*Ky):]
    RERopt=opt.fun
    Gopt=R@wopt+bopt
    Aopt=softmax(Gopt,axis=1)
    # Hessian components (original paper omits division by N)
    d2Ldw2=(1/N)*np.square(R).T@(Aopt-np.square(Aopt))
    d2Ldb2=(1/N)*np.sum(Aopt-np.square(Aopt),axis=0)
    TrHess=np.sum(d2Ldw2)+np.sum(d2Ldb2)
    return RERopt+(Ky*Dr/(2*L))*np.log(1+(L*v0/(Ky*Dr))*TrHess)

#%%

# Testing equivalence of PACTran score functions

N = 100
Ky = 4
Df = 6

F = np.array(np.random.randint(2,size=(N,Df)),dtype=np.float64)
Y = np.random.randint(Ky,size=N)

print(f"F[:10]: \n{F[:10]}")
print(f"Y[:10]: \n{Y[:10]}")
print()

t0 = datetime.datetime.now()
print(f"Original: {PACTran(F,Y,lda_factor=10)[0][4][1]}")
t1 = datetime.datetime.now()
print((t1-t0))
print()
t0 = datetime.datetime.now()
print(f"Verbose: {PACTran_verbose(F,Y)}")
t1 = datetime.datetime.now()
print((t1-t0))
print()
t0 = datetime.datetime.now()
print(f"Concise {PACTran_concise(F,Y)}")
t1 = datetime.datetime.now()
print((t1-t0))

#%%
