


import os

import numpy as np
import matplotlib.pyplot as plt

import keras


reconstructed_model = keras.models.load_model("my_model.keras")
# weights = keras.layers.Layer.get_weights(reconstructed_model)
# print(weights)

checkpoint_dir = 'data/testing_data/nn_weights_01/'

# Load weights and plot
weights_data = []
epochs = np.arange(1,11)

# print(epochs)

for epoch in epochs:
    # print(epoch)
    # Load weights for the current epoch
    weights_path = os.path.join(checkpoint_dir, f'weights_{epoch:02d}.weights.h5')
    reconstructed_model.load_weights(weights_path)

