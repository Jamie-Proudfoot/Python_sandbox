#%%

import numpy as np
import warnings
import datetime
from numba import njit
np.random.seed(0)
print()
#%%

# Original code. Author: youkaichao. Github: https://github.com/thuml/LogME/blob/main/LogME.py
# Modified for functional programming (removing functions from class structure)
# Additional comments added here

@njit # possible error with numpy version
def each_evidence(y_, F, fh, v, s, vh, N, Df):
    """
    compute the maximum evidence for each class
    """
    epsilon = 1e-5
    alpha = 1.0
    beta = 1.0
    lam = alpha / beta
    tmp = (vh @ (F @ np.ascontiguousarray(y_)))
    for _ in range(11):
        # should converge after at most 10 steps
        # typically converge after two or three steps
        gamma = (s / (s + lam)).sum()
        # A = v @ np.diag(alpha + beta * s) @ v.transpose() # no need to compute A
        # A_inv = v @ np.diag(1.0 / (alpha + beta * s)) @ v.transpose() # no need to compute A_inv
        m = v @ (tmp * beta / (alpha + beta * s))
        alpha_de = (m * m).sum()
        alpha = gamma / (alpha_de + epsilon)
        beta_de = ((y_ - fh @ m) ** 2).sum()
        beta = (N - gamma) / (beta_de + epsilon)
        new_lam = alpha / beta
        if np.abs(new_lam - lam) / lam < 0.01:
            break
        lam = new_lam
    evidence = Df / 2.0 * np.log(alpha) \
               + N / 2.0 * np.log(beta) \
               - 0.5 * np.sum(np.log(alpha + beta * s)) \
               - beta / 2.0 * (beta_de + epsilon) \
               - alpha / 2.0 * (alpha_de + epsilon) \
               - N / 2.0 * np.log(2 * np.pi)
    return evidence / N, alpha, beta, m
# use pseudo data to compile the function
# Df = 20, N = 50
f_tmp = np.random.randn(20, 50).astype(np.float64)
each_evidence(np.random.randint(0, 2, 50).astype(np.float64), f_tmp, f_tmp.transpose(), np.eye(20, dtype=np.float64), np.ones(20, dtype=np.float64), np.eye(20, dtype=np.float64), 50, 20)

@njit
def truncated_svd(x):
    u, s, vh = np.linalg.svd(x.transpose() @ x)
    s = np.sqrt(s)
    u_times_sigma = x @ vh.transpose()
    R = np.sum((s > 1e-10) * 1)  # rank of F
    s = s.reshape(-1, 1)
    s = s[:R]
    vh = vh[:R]
    u = u_times_sigma[:, :R] / s.reshape(1, -1)
    return u, s, vh
truncated_svd(np.random.randn(20, 10).astype(np.float64))

class LogME(object):
    def __init__(self, regression=False):
        """
            :param regression: whether regression
        """
        self.regression = regression
        self.fitted = False
        self.reset()

    def reset(self):
        self.num_dim = 0
        self.alphas = []  # alpha for each class / dimension
        self.betas = []  # beta for each class / dimension
        # self.ms.shape --> [C, Df]
        self.ms = []  # m for each class / dimension

    def _fit_fixed_point(self, F: np.ndarray, y: np.ndarray):
        """
        LogME calculation proposed in the arxiv 2021 paper
        "Ranking and Tuning Pre-trained Models: A New Paradigm of Exploiting Model Hubs"
        at https://arxiv.org/abs/2110.10545
        """
        N, Df = F.shape  # R = min(N, Df)
        if N > Df: # direct SVD may be expensive
            u, s, vh = truncated_svd(F)
        else:
            u, s, vh = np.linalg.svd(F, full_matrices=False)
        # u.shape = N x R
        # s.shape = R
        # vh.shape = R x Df
        s = s.reshape(-1, 1)
        sigma = (s ** 2)

        evidences = []
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            y_ = y_.reshape(-1, 1)
            x = u.T @ y_  # x has shape [R, 1], but actually x should have shape [N, 1]
            x2 = x ** 2
            res_x2 = (y_ ** 2).sum() - x2.sum()  # if R < N, we compute sum of xi for 0 singular values directly

            alpha, beta = 1.0, 1.0
            for _ in range(11):
                t = alpha / beta
                gamma = (sigma / (sigma + t)).sum()
                m2 = (sigma * x2 / ((t + sigma) ** 2)).sum()
                res2 = (x2 / ((1 + sigma / t) ** 2)).sum() + res_x2
                alpha = gamma / (m2 + 1e-5)
                beta = (N - gamma) / (res2 + 1e-5)
                t_ = alpha / beta
                evidence = Df / 2.0 * np.log(alpha) \
                           + N / 2.0 * np.log(beta) \
                           - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                           - beta / 2.0 * res2 \
                           - alpha / 2.0 * m2 \
                           - N / 2.0 * np.log(2 * np.pi)
                evidence /= N
                if abs(t_ - t) / t <= 1e-3:  # abs(t_ - t) <= 1e-5 or abs(1 / t_ - 1 / t) <= 1e-5:
                    break
            evidence = Df / 2.0 * np.log(alpha) \
                       + N / 2.0 * np.log(beta) \
                       - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                       - beta / 2.0 * res2 \
                       - alpha / 2.0 * m2 \
                       - N / 2.0 * np.log(2 * np.pi)
            evidence /= N
            m = 1.0 / (t + sigma) * s * x
            m = (vh.T @ m).reshape(-1)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.ms.append(m)
        self.ms = np.stack(self.ms)
        return np.mean(evidences)

    _fit = _fit_fixed_point

    def fit(self, F: np.ndarray, y: np.ndarray):
        """
        :param F: [N, Df], feature matrix from pre-trained model
        :param y: target labels.
            For classification, y has shape [N] with element in [0, C_t).
            For regression, y has shape [N, C] with C regression-labels

        :return: LogME score (how well F can fit y directly)
        """
        if self.fitted:
            warnings.warn('re-fitting for new data. old parameters cleared.')
            self.reset()
        else:
            self.fitted = True
        F = F.astype(np.float64)
        if self.regression:
            y = y.astype(np.float64)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
        return self._fit(F, y)

    def predict(self, F: np.ndarray):
        """
        :param F: [N, Df], feature matrix
        :return: prediction, return shape [N, X]
        """
        if not self.fitted:
            raise RuntimeError("not fitted, please call fit first")
        F = F.astype(np.float64)
        logits = F @ self.ms.T
        if self.regression:
            return logits
        return np.argmax(logits, axis=-1)

print()
#%% 

# Modified code. Author: JamieProudfoot
# Verbose version with commentary
# Based on code by Author: youkaicho from https://github.com/thuml/LogME/blob/main/LogME.py

# Helper functions

# Compute evidence
@njit
def calc_evidence(N, Df, alpha, beta, sigma, res2, m2):
    # Compute log evidence according to eqn (2) in You et al. 2021
    evidence = Df/2*np.log(alpha) \
            + N/2*np.log(beta) \
            - 0.5*np.sum(np.log(alpha+beta*sigma)) \
            - beta/2*res2 \
            - alpha/2*m2 \
            -  N/2*np.log(2*np.pi)
    # Normalise log evidence by N
    evidence /= N
    return evidence
# Precompile with numba for speed
calc_evidence(50,20,1,1,np.random.randn(50)**2,1,1)
print()
#%%

# Compute maximum evidence by optimisation
@njit
def max_evidence(z, Y_, sigma, N, Df):
    """
    Compute the maximum evidence for each class
    """
    z2 = z**2
    Y2 = Y_**2
    # Residual of sum of squares
    res_z2 = np.sum(Y2) - np.sum(z2)
    # Initialise alpha and beta variance
    alpha, beta = 1.0, 1.0
    # Rapidly converges (by at most 10 steps)
    for _ in range(11):
        t = alpha/beta
        gamma = np.sum(sigma/(sigma+t))
        m2 = np.sum(sigma*z2/(sigma+t)**2)
        res2 = np.sum(z2/(1+sigma/t)**2)+res_z2
        # Update alpha and beta variance
        alpha = gamma/(m2+1e-5)
        beta = (N-gamma)/(res2+1e-5)
        # Update alpha beta ratio t
        t_ = alpha/beta
        # Compute evidence
        evidence = calc_evidence(N, Df, alpha, beta, sigma, res2, m2)
        # Early stopping condition for convergence
        if abs(t_-t)/t <= 1e-5: break
    return evidence
# Precompile with numba for speed
max_evidence(np.random.randn(50,1),np.random.randn(50),np.random.randn(50)**2,20,5)
print()
#%%

# Truncated singular value decomposition
@njit
def trunc_svd(x):
    u, s, vh = np.linalg.svd(x.T@x)
    s = np.sqrt(s)
    u_times_sigma = x@vh.T
    # R is the rank of F
    R = np.sum(s > 1e-10)
    s = s.reshape(-1, 1)
    s = s[:R]
    vh = vh[:R]
    u = u_times_sigma[:,:R]/s.reshape(1, -1)
    # Retain only the kth most important values
    return u, s, vh
# Precompile with numba for speed
trunc_svd(np.random.randn(50, 20))
print()
#%%

# Main LogME calculation

def LogME_succinct(F, Y, regression=False):
    """
    LogME calculation proposed in the arxiv 2021 paper
    "Ranking and Tuning Pre-trained Models: A New Paradigm of Exploiting Model Hubs"
    at https://arxiv.org/abs/2110.10545
    F :: 'feature function' (N*Df matrix)
    Y :: Target label (N*r matrix if regression, N vector if classification)
    succinct version
    """
    N,Df=F.shape  
    u,s,vh=trunc_svd(F) if N > Df else np.linalg.svd(F,full_matrices=False)
    s=s.reshape(-1,1)
    sigma=s**2
    evidences=[]
    num_dim=Y.shape[1] if regression else int(np.max(Y)+1)
    for i in range(num_dim):
        Y_=Y[:,i] if regression else (Y==i).astype(np.float64)
        Y_=Y_.reshape(-1,1)
        z=u.T@Y_
        evidence=max_evidence(z,Y_,sigma,N,Df)
        evidences.append(evidence)
    return np.mean(evidences)

#%%
# Testing the equivalence of LogME calculations

# F = np.random.randn(50,20)
# Yc = np.random.randint(2,size=50)
# Yr = np.random.randn(50,1)

# logme = LogME(regression=False)
# print(f"Classification original: {logme.fit(F,Yc)}")
# print(f"Classification succinct: {LogME_succinct(F,Yc)}")
# logme = LogME(regression=True)
# print(f"Regression original: {logme.fit(F,Yr)}")
# print(f"Regression succinct: {LogME_succinct(F,Yr,regression=True)}")
# print()
#%%
