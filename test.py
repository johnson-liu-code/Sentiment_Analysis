import os

import numpy as np
# import matplotlib.pyplot as plt

import keras
from keras.utils import plot_model
import visualkeras

# from PIL import ImageFont

reconstructed_model = keras.models.load_model("my_model.keras")
# weights = keras.layers.Layer.get_weights(reconstructed_model)
# print(weights)

'''
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


weights = reconstructed_model.get_weights()
# print(weights)
# Save the weights to a file.
np.savez('weights.npz', *weights)
'''

weights = np.load('weights.npz', allow_pickle=True)
# Print the weights.
# print(weights.files)
# print(type(weights))
# for i in range(len(weights.files)):
    # print(f'weights[{i}]:\n{weights[weights.files[i]]}')
    # print(f'weights[{i}].shape: {weights[weights.files[i]].shape}')
    # print(f'weights[{i}].dtype: {weights[weights.files[i]].dtype}')
    # print(f'weights[{i}].ndim: {weights[weights.files[i]].ndim}')
    # print(f'weights[{i}].itemsize: {weights[weights.files[i]].itemsize}')


# plot_model(reconstructed_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# reconstructed_model.summary()

# from keras import Model, Input
# from keras import layers
# import visualkeras
# from PIL import ImageFont

# Define the model architecture
# input1 = Input(shape=(8,), name="InputLayer")
# dense1 = layers.Dense(4, name="DenseLayer1")(input1)
# concat = layers.Concatenate(name="ConcatenateLayer")([dense1, input1])
# output = layers.Dense(1, name="OutputLayer")(concat)

# Create the model
# model = Model(inputs=input1, outputs=output, name="ConcatenateExampleModel")

# Visualize the model
# font = ImageFont.truetype("arial.ttf", 12)
# visualkeras.layered_view(model, legend=True, font=font).save('fancy_model_plot.png')