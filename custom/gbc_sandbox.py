#%%

import numpy as np
import datetime
import tensorflow as tf

from sklearn.decomposition import PCA

#%%

# Original code. Author: Google-Research. 

# Github: https://github.com/google-research/google-research/blob/b7cb09ba04bf018f1ba1131ccd4c69be617e8e2f/stable_transfer/transferability/gbc.py#L27

# Comments added here. Author: JamieProudfoot.

def compute_bhattacharyya_distance(mu1, mu2, sigma1, sigma2):
  """Compute Bhattacharyya distance between diagonal or spherical Gaussians."""
  # Average variance as described in equations of GBC
  avg_sigma = (sigma1 + sigma2) / 2
  # Compute DB (Bhattacharyya distance) as described in equations of GBC
  first_part = tf.reduce_sum((mu1 - mu2)**2 / avg_sigma) / 8
  second_part = tf.reduce_sum(tf.math.log(avg_sigma))
  second_part -= 0.5 * (tf.reduce_sum(tf.math.log(sigma1)))
  second_part -= 0.5 * (tf.reduce_sum(tf.math.log(sigma2)))
  return first_part + 0.5 * second_part


def get_bhattacharyya_distance(per_class_stats, c1, c2, gaussian_type):
  """Return Bhattacharyya distance between 2 diagonal or spherical gaussians."""
  mu1 = per_class_stats[c1]['mean']
  mu2 = per_class_stats[c2]['mean']
  sigma1 = per_class_stats[c1]['variance']
  sigma2 = per_class_stats[c2]['variance']
  # Spherical Gaussians => scalar variance
  # Diagonal Gaussians => vector variance
  if gaussian_type == 'spherical':
    sigma1 = tf.reduce_mean(sigma1)
    sigma2 = tf.reduce_mean(sigma2)
  return compute_bhattacharyya_distance(mu1, mu2, sigma1, sigma2)


def compute_per_class_mean_and_variance(features, target_labels, unique_labels):
  """Compute features mean and variance for each class."""
  per_class_stats = {}
  for label in unique_labels:
    label = int(label)  # For correct indexing
    per_class_stats[label] = {}
    class_ids = tf.equal(target_labels, label)
    class_features = tf.gather_nd(features, tf.where(class_ids))
    # Compute per-class feature mean
    mean = tf.reduce_mean(class_features, axis=0)
    # Compute per-class feature variance (ddof=0)
    variance = tf.math.reduce_variance(class_features, axis=0)
    per_class_stats[label]['mean'] = mean
    # Avoid 0 variance in cases of constant features with tf.maximum
    per_class_stats[label]['variance'] = tf.maximum(variance, 1e-4)
    # per_class_stats[label]['variance'] = variance
  return per_class_stats


def GBC(features, target_labels, gaussian_type, p=0.9):
  """Compute Gaussian Bhattacharyya Coefficient (GBC).

  Args:
    features: source features from the target data.
    target_labels: ground truth labels in the target label space.
    gaussian_type: type of gaussian used to represent class features. The
      possibilities are spherical (default) or diagonal.

  Returns:
    gbc: transferability metric score.
  """
  assert gaussian_type in ('diagonal', 'spherical')

  # PCA reduction of features added here (modified)
  features = PCA(n_components=int(np.rint(features.shape[1]*p))).fit_transform(features)
  # List of all classes
  unique_labels, _ = tf.unique(target_labels)
  unique_labels = list(unique_labels)
  # Compute per-class stats (mean and variance)
  per_class_stats = compute_per_class_mean_and_variance(
      features, target_labels, unique_labels)
  # Loop over pairs of classes c1 != c2
  per_class_bhattacharyya_distance = []
  for c1 in unique_labels:
    temp_metric = []
    for c2 in unique_labels:
      if c1 != c2:
        # Compute DB (Bhattacharyya Distance) for c1 and c2
        bhattacharyya_distance = get_bhattacharyya_distance(
            per_class_stats, int(c1), int(c2), gaussian_type)
        # Compute BC (Bhattacharyya Coefficient) from DB and append
        temp_metric.append(tf.exp(-bhattacharyya_distance))
    # Compute sum of BC for each class
    per_class_bhattacharyya_distance.append(tf.reduce_sum(temp_metric))
  # Compute -ve sum of BC for all classes
  gbc = -tf.reduce_sum(per_class_bhattacharyya_distance)

  return gbc


#%%

# Verbose code. Author: JamieProudfoot.

# Helper function for per-class sample statistics
def class_stats(X, Y, c):
    """
    X :: features (N*D matrix)
    Y :: labels (N matrix)
    Compute sample statistics of all X in a given class c
    returns :: mu_c, var_c
    where mu_c is the sample mean of all X in class c
    and var_c is the sample variance of all X in class c
    """
    Xc = X[Y==c]
    # Original code uses ddof=0 (no Bessel correction)
    return (np.mean(Xc,axis=0),np.maximum(np.var(Xc,ddof=1,axis=0),1e-4))

# Helper function for Gaussian Bhattacharyya Coefficient
def BC(mu_cl, var_cl, mu_ck, var_ck):
    """
    X :: features (N*D matrix)
    Y :: labels (N matrix)
    Compute Gaussian Bhattacharyya Coefficient
    between two classes cl and ck
    returns :: gbc_lk (Gaussian Bhattacharyya Coefficient between cl and ck)
    """
    # Averaged class variance
    var = 0.5*(var_cl+var_ck)
    # Bhattacharyya Distance
    DB_lk = np.sum(0.125*(mu_cl-mu_ck)**2/var \
        + 0.5*np.log(var) \
        - 0.25*np.log(var_ck) \
        - 0.25*np.log(var_cl))
    return np.exp(-DB_lk)

def GBC_verbose(F, Y, p=0.9):
    """
    F :: 'feature function' (N*Df matrix)
    Y :: Target label (N*r matrix if regression, N vector if classification)
    p :: Dr/Df where Dr is the PCA-reduced dimensionality [0.5, 0.9]
    returns :: GBC (Gaussian Bhattacharyya Coefficient) measure to
    predict the success of transfer learning
    verbose version
    """
    # N: number of data points, Df: dimensionality of hidden representation
    N, Df = F.shape
    # Ky: number of target label classes
    Ky = int(Y.max() + 1)
    # PCA-reduced dimensions
    Dr = int(np.rint(Df*p))
    # PCA-reduced representation R
    pca = PCA(n_components=Dr)
    R = pca.fit_transform(F)
    # Compute class statistics
    S = [class_stats(R,Y,c) for c in range(Ky)]
    # Pairs of classes cl != ck without repeats
    triu = np.transpose(np.triu_indices(Ky,1))
    # Array of Bhattacharyya Coefficients
    gbc = [BC(*S[t[0]],*S[t[1]]) for t in triu]
    return -2*sum(gbc)

#%%

# Succinct code. Author: JamieProudfoot.

def GBC_succinct(F,Y,p=0.9):
    """
    F :: 'feature function' (N*Df matrix)
    Y :: Target label (N*r matrix if regression, N vector if classification)
    p :: Dr/Df where Dr is the PCA-reduced dimensionality [0.5, 0.9]
    returns :: GBC (Gaussian Bhattacharyya Coefficient) measure to
    predict the success of transfer learning
    succinct version
    """
    Ky=int(Y.max()+1)
    R=PCA(n_components=int(np.rint(F.shape[1]*p))).fit_transform(F)
    S = [class_stats(R,Y,c) for c in range(Ky)]
    return -2*sum([BC(*S[t[0]],*S[t[1]]) for t in np.transpose(np.triu_indices(Ky,1))])

#%%

# Testing equivalence of GBC score functions

# N = 100
# Ky = 4
# Df = 5

# F = np.array(np.random.randint(2,size=(N,Df)),dtype=np.float64)
# Y = np.random.randint(Ky,size=N)

# print(f"F[:10]:\n{F[:10]}")
# print(f"Y[:10]:\n{Y[:10]}")
# print()

# t0 = datetime.datetime.now()
# print(f"Original: {GBC(F,Y,'diagonal')}")
# t1 = datetime.datetime.now()
# print((t1-t0))
# print()
# t0 = datetime.datetime.now()
# print(f"Verbose: {GBC_verbose(F,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))
# print()
# t0 = datetime.datetime.now()
# print(f"Succinct: {GBC_succinct(F,Y)}")
# t1 = datetime.datetime.now()
# print((t1-t0))

#%%