#%%

import numpy as np
import datetime

#%%

# Original code. Author: youkaichao. Github: https://github.com/thuml/LogME/blob/main/LEEP.py

def LEEP(A: np.ndarray, Y: np.ndarray):
    """

    :param pseudo_source_label: shape [N, C_s]
    :param target_label: shape [N], elements in [0, C_t)
    :return: leep score
    """
    # added for consistent notation
    pseudo_source_label = A
    target_label = Y
    # added for consistent notation
    N, C_s = pseudo_source_label.shape
    target_label = target_label.reshape(-1)
    C_t = int(np.max(target_label) + 1)   # the number of target classes (assuming max(target_label) == C_t)
    normalized_prob = pseudo_source_label / float(N)  # sum(normalized_prob) = 1
    joint = np.zeros((C_t, C_s), dtype=float)  # placeholder for joint distribution over (y, z)
    for i in range(C_t):
        this_class = normalized_prob[target_label == i]
        row = np.sum(this_class, axis=0)
        joint[i] = row
    p_target_given_source = (joint / joint.sum(axis=0, keepdims=True)).T  # P(y | z)

    empirical_prediction = pseudo_source_label @ p_target_given_source # TH_z(x) @ P(y | z) (matrix multiplication)
    empirical_prob = np.array([predict[label] for predict, label in zip(empirical_prediction, target_label)])
    leep_score = np.mean(np.log(empirical_prob))
    return leep_score

#%%

# Modified code (verbose). Author: JamieProudfoot.

def LEEP_verbose(A,Y):
    """
    A :: pseudo source label (N*Kz matrix)
    Y :: target label (N vector)
    returns leep :: numerical meausure of source model 
    to target dataset transferability
    verbose version
    """
    # N: number of data points, Kz: number of pseudo source label classes
    N, Kz = A.shape
    # Normalise Z so that total sum over Z == 1
    normalised_A = A / N
    # Ky: number of target label classes
    Ky=int(np.max(Y) + 1)
    # Compute joint probability matrix P(y,z)
    joint = (Y == np.vstack(range(Ky)))@normalised_A
    # Compute conditional probability matrix P(y|z) = P(y,z) / P(z)
    conditional = (joint / joint.sum(axis=0)).T
    # Compute EEP (expected empirical prediction)
    marginal = A @ conditional
    EEP = np.array([py[y] for py, y in zip(marginal, Y)])
    # Compute LEEP (log of expected empirical prediction)
    leep = np.mean(np.log(EEP))
    return leep

#%%

# Modified code (succinct). Author: JamieProudfoot.

def LEEP_succinct(A,Y):
    """
    A :: pseudo source label (N*Kz matrix)
    Y :: target label (N vector)
    returns leep :: numerical meausure of source model 
    to target dataset transferability
    succinct version
    """
    N,Kz=A.shape
    An=A/N
    Ky=int(np.max(Y)+1)
    joint=(Y==np.vstack(range(Ky)))@An
    conditional=(joint/joint.sum(axis=0)).T
    marginal=A@conditional
    EEP=np.array([py[y] for py,y in zip(marginal,Y)])
    return np.mean(np.log(EEP))

#%%

# Testing equivalence of LEEP score functions

# A = np.array([[0.3,0.7],[0.2,0.8],[0.9,0.1],[0.55,0.45],[0.7,0.3]])
# Y = np.array([1,1,0,0,1])

# print(f"A:\n{A}")
# print(f"Y:\n{Y}")
# print()

# t0 = datetime.datetime.now()
# print(f"Original: {LEEP(A,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))
# print()
# t0 = datetime.datetime.now()
# print(f"Verbose: {LEEP_verbose(A,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))
# print()
# t0 = datetime.datetime.now()
# print(f"Succinct: {LEEP_succinct(A,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))

#%%
