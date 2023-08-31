#%%

import numpy as np
import tensorflow as tf
from tensorflow import keras

#%%
# Number of classes
K = 3

print("Feature extraction from one layer only")
initial_model = keras.Sequential(
   [
      keras.Input(shape=(250, 250, 3)),
      keras.layers.Conv2D(32, 5, strides=2, activation="relu"),
      keras.layers.MaxPooling2D((4,4)),
      keras.layers.Conv2D(32, 2, activation="relu"),
      keras.layers.MaxPooling2D((4,4)),
      keras.layers.Conv2D(16, 2, activation="relu", name="my_intermediate_layer"),
      keras.layers.MaxPooling2D((3,3)),
      keras.layers.Conv2D(16, 2, activation="relu"),
      keras.layers.Flatten(),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(K, activation="softmax")
   ]
)

print("Feature extraction from the model")
feature_extractor = keras.Model(
   inputs=initial_model.inputs,
   outputs=keras.layers.Flatten()(initial_model.get_layer(name="my_intermediate_layer").output),
)

#%% 

print("The feature extractor method is called on test data")
N = 600
D = 250
# One 250x250 RGB image as input data
X = np.random.randn(N, D, D, K)
# X = np.ones((N, D, D, 3))
X = tf.convert_to_tensor(X)
print(np.array(X).shape)
# One classification label as output data
Y = np.random.randint(3, size=(1,N))
Y = Y.flatten()
print(Y)
# Dummy classification labels
A = initial_model.predict(X)
print(A)
Z = A.argmax(axis=-1)
print(Z)
# Extracted features for this image
features = feature_extractor(X)
F = np.array(features).reshape(N,-1)
print(F.shape)

#%%
# H-Score

from hscore_sandbox import HScore, HScore_verbose, HScore_succinct, HScore_regularised
print(f'Original: {HScore(F,Y)}')
print(f'Verbose: {HScore_verbose(F,Y)}')
print(f'Succinct: {HScore_succinct(F,Y)}')
print(f'Regularised: {HScore_regularised(F,Y)}')

#%%
# NCE

from nce_sandbox import NCE, NCE_verbose, NCE_succinct
print(f'Original: {NCE(Z,Y)}')
print(f'Verbose: {NCE_verbose(Z,Y)}')
print(f'Succinct: {NCE_succinct(Z,Y)}')

#%%
# LEEP

from leep_sandbox import LEEP, LEEP_verbose, LEEP_succinct
print(f'Original: {LEEP(A,Y)}')
print(f'Verbose: {LEEP_verbose(A,Y)}')
print(f'Succinct: {LEEP_succinct(A,Y)}')

#%%
# LogME

from logme_sandbox import LogME, LogME_succinct
logme = LogME(regression=False)
print(f'Original: {logme.fit(F,Y)}')
print(f'Succinct: {LogME_succinct(F,Y)}')

#%%
# N-LEEP

from nleep_sandbox import NLEEP_succinct
logme = LogME(regression=False)
print(f'Succinct: {NLEEP_succinct(F,Y)}')

#%%