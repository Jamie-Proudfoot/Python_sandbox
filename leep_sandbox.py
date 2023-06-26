#%%

import numpy as np
import datetime

#%%

# Original code. Author: youkaichao. Github: https://github.com/thuml/LogME/blob/main/LEEP.py

def LEEP(Z: np.ndarray, Y: np.ndarray):
    """

    :param pseudo_source_label: shape [N, C_s]
    :param target_label: shape [N], elements in [0, C_t)
    :return: leep score
    """
    # added for consistent notation
    pseudo_source_label = Z
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

def LEEP_concise(Z,Y):
    """
    Z :: pseudo source label (n*k_z matrix)
    Y :: target label (n vector)
    returns leep :: numerical meausure of source model 
    to target dataset transferability
    concise version
    """
    # n: number of data points, k_z: number of pseudo source label classes
    n, k_z = Z.shape
    # Normalise Z so that total sum over Z == 1
    normalised_Z=Z/n
    # k_y: number of target label classes
    k_y=int(np.max(Y) + 1)
    # Initialise joint probability matrix P(y,z)
    joint=np.zeros(k_y,k_z)
    # Single loop over target data classes
    for y in range(k_y): joint[y] = np.sum(normalised_Z[Y==y])
    # Compute conditional probability matrix P(y|z) = P(y,z) / P(z)
    conditional = (joint / joint.sum(axis=0, keepdims=True)).T
    # Compute EEP (expected empirical prediction)
    marginal = Z @ conditional
    EEP = np.array([py[y] for py, y in zip(marginal, Y)])
    # Compute LEEP (log of expected empirical prediction)
    leep = np.mean(np.log(EEP))
    return leep

#%%

# Modified code (succinct). Author: JamieProudfoot.

def LEEP_concise(Z,Y):
    """
    Z :: pseudo source label (n*k_z matrix)
    Y :: target label (n vector)
    returns leep :: numerical meausure of source model 
    to target dataset transferability
    concise version
    """
    n,k_z=Z.shape
    normalised_Z=Z/n
    k_y=int(np.max(Y)+1)
    joint=np.zeros(k_y,k_z)
    for y in range(k_y): joint[y]=np.sum(normalised_Z[Y==y])
    conditional=(joint/joint.sum(axis=0,keepdims=True)).T
    marginal=Z@conditional
    EEP=np.array([py[y] for py,y in zip(marginal,Y)])
    return np.mean(np.log(EEP))

#%%