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
def each_evidence(y_, f, fh, v, s, vh, N, D):
    """
    compute the maximum evidence for each class
    """
    epsilon = 1e-5
    alpha, beta = 1.0, 1.0
    lam = alpha / beta
    tmp = (vh @ (f @ np.ascontiguousarray(y_)))
    for _ in range(11):
        gamma = (s / (s + lam)).sum()
        m = v @ (tmp * beta / (alpha + beta * s))
        alpha_de = (m * m).sum()
        alpha = gamma / (alpha_de + epsilon)
        beta_de = ((y_ - fh @ m) ** 2).sum()
        beta = (N - gamma) / (beta_de + epsilon)
        new_lam = alpha / beta
        # Early stopping condition for convergence
        if np.abs(new_lam - lam) / lam < 0.001:
            break
        lam = new_lam
    evidence = D / 2.0 * np.log(alpha) \
               + N / 2.0 * np.log(beta) \
               - 0.5 * np.sum(np.log(alpha + beta * s)) \
               - beta / 2.0 * (beta_de + epsilon) \
               - alpha / 2.0 * (alpha_de + epsilon) \
               - N / 2.0 * np.log(2 * np.pi)
    return evidence / N, alpha, beta, m
# Use pseudo data to compile the function
# n = 50, d = 20
f_tmp = np.random.randn(20, 50).astype(np.float64)
each_evidence(np.random.randint(0, 2, 50).astype(np.float64), f_tmp, f_tmp.transpose(), np.eye(20, dtype=np.float64), np.ones(20, dtype=np.float64), np.eye(20, dtype=np.float64), 50, 20)


@njit
def each_evidence(Y_, x, sigma, n, d):
    """
    Compute the maximum evidence for each class
    """
    x2 = x ** 2
    res_x2 = (Y_ ** 2).sum() - x2.sum()
    alpha, beta = 1.0, 1.0
    eps = 1e-5
    for _ in range(11):
        t = alpha / beta
        gamma = (sigma / (sigma + t)).sum()
        m2 = (sigma * x2 / ((t + sigma) ** 2)).sum()
        res2 = (x2 / ((1 + sigma / t) ** 2)).sum() + res_x2
        alpha = gamma / (m2 + eps)
        beta = (n - gamma) / (res2 + eps)
        t_ = alpha / beta
        evidence = d / 2.0 * np.log(alpha) \
                + n / 2.0 * np.log(beta) \
                - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                - beta / 2.0 * res2 \
                - alpha / 2.0 * m2 \
                - n / 2.0 * np.log(2 * np.pi)
        evidence /= n
        # Early stopping condition for convergence
        if abs(t_ - t) / t <= 1e-3:  # abs(t_ - t) <= 1e-5 or abs(1 / t_ - 1 / t) <= 1e-5:
            break
    evidence = d / 2.0 * np.log(alpha) \
            + n / 2.0 * np.log(beta) \
            - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
            - beta / 2.0 * res2 \
            - alpha / 2.0 * m2 \
            - n / 2.0 * np.log(2 * np.pi)
    evidence /= n
    return evidence
# Use pseudo data to compile the function
# n = 50, d = 20
f_tmp = np.random.randn(20, 50).astype(np.float64)
each_evidence(np.random.randint(0, 2, 50).astype(np.float64), f_tmp, f_tmp.transpose(), np.eye(20, dtype=np.float64), np.ones(20, dtype=np.float64), np.eye(20, dtype=np.float64), 50, 20)


@njit
def truncated_svd(x):
    u, s, vh = np.linalg.svd(x.transpose() @ x)
    s = np.sqrt(s)
    u_times_sigma = x @ vh.transpose()
    k = np.sum((s > 1e-10) * 1)  # rank of f
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
    n, d = f.shape  # k = min(n, d)
    if n > d: # direct SVD may be expensive
        u, s, vh = truncated_svd(f)
    else:
        u, s, vh = np.linalg.svd(f, full_matrices=False)
    s = s.reshape(-1, 1)
    sigma = (s ** 2)
    evidences = []
    num_dim = Y.shape[1] if regression else int(Y.max()+1)
    for i in range(num_dim):
        Y_ = Y[:, i] if regression else (Y == i).astype(np.float64)
        Y_ = Y_.reshape(-1, 1)
        x = u.T @ Y_  # x has shape [k, 1], but actually x should have shape [N, 1]
        evidence = each_evidence(Y_, x, sigma, n, d)
        evidences.append(evidence)
    return np.mean(evidences)