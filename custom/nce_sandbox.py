#%%

import numpy as np
import datetime

# In case some pseudo-classes are unoccupied and p_z = 0
np.seterr(divide='ignore', invalid='ignore')

#%%

# Original code. Author: youkaichao. Github: https://github.com/thuml/LogME/blob/main/NCE.py


def NCE(Z: np.ndarray, Y: np.ndarray):
    """

    :param source_label: shape [N], elements in [0, C_s), often got from taking argmax from pre-trained predictions
    :param target_label: shape [N], elements in [0, C_t)
    :return:
    """
    source_label = Z
    target_label = Y
    C_t = int(np.max(target_label) + 1)  # the number of target classes
    C_s = int(np.max(source_label) + 1)  # the number of source classes
    N = len(source_label)
    joint = np.zeros((C_t, C_s), dtype=float)  # placeholder for the joint distribution, shape [C_t, C_s]
    for s, t in zip(source_label, target_label):
        s = int(s)
        t = int(t)
        joint[t, s] += 1.0 / N
    p_z = joint.sum(axis=0, keepdims=True)  # shape [1, C_s]
    p_target_given_source = (joint / p_z).T  # P(y | z), shape [C_s, C_t]
    mask = p_z.reshape(-1) != 0  # valid Z, shape [C_s]
    p_target_given_source = p_target_given_source[mask] + 1e-20  # remove NaN where p(z) = 0, add 1e-20 to avoid log (0)
    entropy_y_given_z = np.sum(- p_target_given_source * np.log(p_target_given_source), axis=1, keepdims=True)  # shape [C_s, 1]
    conditional_entropy = np.sum(entropy_y_given_z * p_z.reshape((-1, 1))[mask]) # scalar
    return -conditional_entropy

#%%

# Modified code (verbose). Author: JamieProudfoot.

def NCE_verbose(Z,Y):
    """
    Z :: source label (N vector)
    Y :: target label (N vector)
    returns nce :: numerical measure of source dataset
    to target dataset transferability
    verbose version
    """
    # Number of target classes
    K_y = int(np.max(Y) + 1) 
    # Number of source classes
    K_z = int(np.max(Z) + 1)
    N = len(Z)
    # Joint probability of target and source labels P(z,y) (K_z*K_y matrix)
    joint = np.histogram2d(Z, Y, bins = (K_z, K_y))[0] / N
    # Marginal probability of source label P(z) (K_z*1 matrix)
    marginal = joint.sum(axis=1, keepdims=True)
    # Conditional probability matrix P(y|z) = P(z,y) / P(z) (K_z*K_y matrix)
    conditional = np.divide(joint,marginal)
    # Mask variable for boolean where P(z) is not zero (K_z boolean vector)
    mask = marginal.reshape(-1) != 0
    # Remove zero values to avoid log asymptote
    conditional = conditional[mask] + 1e-16
    # Entropy of Y given Z == H(y|z) (K_z*1 matrix)
    H = np.sum(-conditional * np.log(conditional), axis=1, keepdims=True)
    # Conditional entropy; sum only when P(z) is not zero
    ce = np.sum(H * marginal.reshape((-1, 1))[mask])
    # Negative conditional entropy
    nce = -ce
    return nce

#%%

# Modified code (succinct). Author: JamieProudfoot.

def NCE_succinct(Z,Y):
    """
    Z :: source label (N vector)
    Y :: target label (N vector)
    returns nce :: numerical measure of source dataset
    to target dataset transferability
    succinct version
    """
    joint=np.histogram2d(Z,Y,bins=(int(np.max(Z)+1),int(np.max(Y)+1)))[0]/len(Z)
    marginal=joint.sum(axis=1,keepdims=True)
    mask=marginal.reshape(-1)!=0
    conditional=np.divide(joint,marginal)[mask]+1e-16
    H=np.sum(-conditional*np.log(conditional),axis=1,keepdims=True)
    return -np.sum(H*marginal.reshape((-1, 1))[mask])

#%%

# Testing equivalence of NCE score functions

# Z = np.array([0,1,0,2,1,3])
# Y = np.array([0,1,0,2,2,1])

# print(f"Z:\n{Z}")
# print(f"Y:\n{Y}")
# print()

# t0 = datetime.datetime.now()
# print(f"Original: {NCE(Z,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))
# print()
# t0 = datetime.datetime.now()
# print(f"Verbose: {NCE_verbose(Z,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))
# print()
# t0 = datetime.datetime.now()
# print(f"Succinct: {NCE_succinct(Z,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))

#%%