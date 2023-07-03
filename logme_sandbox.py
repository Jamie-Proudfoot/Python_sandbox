#%%

import numpy as np
import datetime
from numba import njit

#%%

# Original code. Author: youkaichao. Github: https://github.com/thuml/LogME/blob/main/LogME.py
# Modified for functional programming (removing functions from class structure)

def LogME(f, Y, regression=False):
    """
    LogME calculation proposed in the arxiv 2021 paper
    "Ranking and Tuning Pre-trained Models: A New Paradigm of Exploiting Model Hubs"
    at https://arxiv.org/abs/2110.10545
    """
    N, D = f.shape  # k = min(N, D)
    if N > D: # direct SVD may be expensive
        u, s, vh = truncated_svd(f)
    else:
        u, s, vh = np.linalg.svd(f, full_matrices=False)
    # u :: (n*k matrix)
    # s ;: (k vector)
    # vh :: (k*d matrix)
    s = s.reshape(-1, 1)
    sigma = (s ** 2)
    evidences = []
    num_dim = Y.shape[1] if regression else int(Y.max() + 1)
    for i in range(num_dim):
        Y_ = Y[:, i] if regression else (Y == i).astype(np.float64)
        Y_ = Y_.reshape(-1, 1)
        x = u.T @ Y_  # x has shape [k, 1], but actually x should have shape [N, 1]
        x2 = x ** 2
        res_x2 = (Y_ ** 2).sum() - x2.sum()  # if k < N, we compute sum of xi for 0 singular values directly
        alpha, beta = 1.0, 1.0
        for _ in range(11):
            t = alpha / beta
            gamma = (sigma / (sigma + t)).sum()
            m2 = (sigma * x2 / ((t + sigma) ** 2)).sum()
            res2 = (x2 / ((1 + sigma / t) ** 2)).sum() + res_x2
            alpha = gamma / (m2 + 1e-5)
            beta = (N - gamma) / (res2 + 1e-5)
            t_ = alpha / beta
            evidence = D / 2.0 * np.log(alpha) \
                    + N / 2.0 * np.log(beta) \
                    - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                    - beta / 2.0 * res2 \
                    - alpha / 2.0 * m2 \
                    - N / 2.0 * np.log(2 * np.pi)
            evidence /= N
            # Early stopping condition for convergence
            if abs(t_ - t) / t <= 1e-3:  # abs(t_ - t) <= 1e-5 or abs(1 / t_ - 1 / t) <= 1e-5:
                break
        evidence = D / 2.0 * np.log(alpha) \
                + N / 2.0 * np.log(beta) \
                - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                - beta / 2.0 * res2 \
                - alpha / 2.0 * m2 \
                - N / 2.0 * np.log(2 * np.pi)
        evidence /= N
        m = 1.0 / (t + sigma) * s * x
        m = (vh.T @ m).reshape(-1)
        evidences.append(evidence)
    return np.mean(evidences)