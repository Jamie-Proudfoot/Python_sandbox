#%%

import numpy as np
import tensorflow as tf
from tensorflow import keras

#%%

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
   ]
)

print("Feature extraction from the model")
feature_extractor = keras.Model(
   inputs=initial_model.inputs,
   outputs=keras.layers.Flatten()(initial_model.get_layer(name="my_intermediate_layer").output),
)

#%% 

print("The feature extractor method is called on test data")
n = 600
d = 250
# One 250x250 RGB image as input data
X = np.random.randn(n, d, d, 3)
# X = np.ones((n, d, d, 3))
X = tf.convert_to_tensor(X)
print(np.array(X).shape)
# One classification label as output data
Y = np.random.randint(3, size=(1,n))
Y = Y.flatten()
print(Y)
# Extracted features for this image
features = feature_extractor(X)
f = np.array(features).reshape(n,-1)
print(f.shape)

#%%

from hscore_sandbox import HScore, HScore_verbose, HScore_succinct, HScore_regularised
print(f'Original: {HScore(f,Y)}')
print(f'Verbose: {HScore_verbose(f,Y)}')
print(f'Succinct: {HScore_succinct(f,Y)}')
print(f'Regularised: {HScore_regularised(f,Y)}')

#%%