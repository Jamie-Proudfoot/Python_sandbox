#%%

import numpy as np
import datetime
from numba import njit
np.random.seed(0)

#%%

# Original code. Author: youkaichao. Github: https://github.com/thuml/LogME/blob/main/LogME.py
# Modified for functional programming (removing functions from class structure)
# Additional comments added here

# Helper functions

@njit
def calc_evidence(n, d, alpha, beta, sigma, res2, m2):
    evidence = d / 2.0 * np.log(alpha) \
            + n / 2.0 * np.log(beta) \
            - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
            - beta / 2.0 * res2 \
            - alpha / 2.0 * m2 \
            -  n / 2.0 * np.log(2 * np.pi)
    evidence /= n
    return evidence

@njit
def max_evidence(x, Y_, sigma, n, d):
    """
    Compute the maximum evidence for each class
    """
    x2 = x**2
    Y2 = Y_**2
    res_x2 = np.sum(Y2) - np.sum(x2)
    alpha, beta = 1.0, 1.0
    for _ in range(11):
        t = alpha/beta
        gamma = np.sum(sigma/(sigma+t))
        m2 = np.sum(sigma*x2/(sigma+t)**2)
        res2 = np.sum(x2/(1+sigma/t)**2)+res_x2
        alpha = gamma/(m2+1e-5)
        beta = (n-gamma)/(res2+1e-5)
        t_ = alpha/beta
        evidence = calc_evidence(n, d, alpha, beta, sigma, res2, m2)
        if abs(t_-t)/t <= 1e-5: break
    return evidence

@njit
def truncated_svd(x):
    u, s, vh = np.linalg.svd(x.transpose() @ x)
    s = np.sqrt(s)
    u_times_sigma = x @ vh.transpose()
    k = np.sum(s > 1e-10)  # rank of f
    s = s.reshape(-1, 1)
    s = s[:k]
    vh = vh[:k]
    u = u_times_sigma[:, :k] / s.reshape(1, -1)
    return u, s, vh
truncated_svd(np.random.randn(20, 10).astype(np.float64))

# Main LogME calculation

def LogME(f, Y, regression=False):
    """
    LogME calculation proposed in the arxiv 2021 paper
    "Ranking and Tuning Pre-trained Models: A New Paradigm of Exploiting Model Hubs"
    at https://arxiv.org/abs/2110.10545
    """
    n, d = f.shape  
    u, s, vh = truncated_svd(f) if n > d else np.linalg.svd(f, full_matrices=False)
    s = s.reshape(-1, 1)
    sigma = s**2
    evidences = []
    num_dim = Y.shape[1] if regression else int(Y.max()+1)
    for i in range(num_dim):
        Y_ = Y[:, i] if regression else (Y == i).astype(np.float64)
        Y_ = Y_.reshape(-1, 1)
        x = u.T @ Y_
        evidence = max_evidence(x, Y_, sigma, n, d)
        evidences.append(evidence)
    return np.mean(evidences)