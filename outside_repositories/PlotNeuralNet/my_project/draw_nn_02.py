import matplotlib.pyplot as plt
from math import cos, sin, atan
import numpy as np
import random




class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        circle = plt.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        plt.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, weights):
        self.previous_layer = self.get_previous_layer(network)
        self.x = self.layer_x_position()  # Horizontal position of the layer
        self.neurons = self.intialise_neurons(number_of_neurons)  # Initialize neurons
        self.weights = weights

    def intialise_neurons(self, number_of_neurons):
        neurons = []
        # Calculate the top margin to center neurons vertically
        y = self.top_margin(number_of_neurons)
        for _ in range(number_of_neurons):
            neuron = Neuron(self.x, y)
            neurons.append(neuron)
            y += vertical_distance_between_neurons  # Space neurons vertically
        return neurons

    def top_margin(self, number_of_neurons):
        # Center neurons vertically within the tallest layer
        return vertical_distance_between_neurons * (number_of_neurons_in_tallest_layer - number_of_neurons) / 2

    def layer_x_position(self):
        if self.previous_layer:
            # Position this layer to the right of the previous layer
            return self.previous_layer.x + horizontal_distance_between_layers
        else:
            # Start at the leftmost position for the first layer
            return 0

    def get_previous_layer(self, network):

        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def line_between_two_neurons(self, neuron1, neuron2, weight):
        # Handle the case where neuron2.y == neuron1.y to avoid division by zero
        if neuron2.y == neuron1.y:
            angle = 0  # Horizontal line
        else:
            # Calculate the angle of the line
            angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))

        # Adjust the start and end points to avoid entering the nodes
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line_x_data = (neuron1.x + x_adjustment, neuron2.x - x_adjustment)
        line_y_data = (neuron1.y + y_adjustment, neuron2.y - y_adjustment)

        # Map the weight to a color (red for negative, blue for positive)
        normalized_weight = (weight + 1) / 2  # Normalize weight to range [0, 1]
        color = (normalized_weight, 0, 1 - normalized_weight)  # RGB: red to blue

        # Draw the line with the calculated color
        line = plt.Line2D(line_x_data, line_y_data, color=color, alpha=0.4, linewidth=1)
        plt.gca().add_line(line)


    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw()
            # Skip drawing lines if there are no weights
            if self.previous_layer and self.weights is not None:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    # Ensure the weights matrix is accessed correctly
                    weight = self.weights[this_layer_neuron_index, previous_layer_neuron_index]
                    self.line_between_two_neurons(neuron, previous_layer_neuron, weight)



class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons, weights=None):
        layer = Layer(self, number_of_neurons, weights)
        self.layers.append(layer)

    def draw(self):
        fig, ax = plt.subplots()

        for layer in self.layers:
            layer.draw()

        plt.axis('scaled')
        plt.xticks([])
        plt.yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.show()


if __name__ == "__main__":
    # Define necessary variables
    vertical_distance_between_layers = 6  # Distance between layers vertically
    horizontal_distance_between_neurons = 2  # Distance between neurons horizontally
    vertical_distance_between_neurons = 2  # Distance between neurons vertically
    horizontal_distance_between_layers = 6  # Distance between layers horizontally
    neuron_radius = 0.5  # Radius of each neuron
    number_of_neurons_in_widest_layer = 4  # Number of neurons in the widest layer
    number_of_neurons_in_tallest_layer = 10  # Number of neurons in the tallest layer

    # Create the neural network
    network = NeuralNetwork()

    # Define weights for the layers
    low = -1
    high = 1
    fromsize = 4  # Number of neurons in the first layer
    tosize = 10  # Number of neurons in the second layer
    weights1 = np.array([[random.uniform(low, high) for _ in range(fromsize)] for _ in range(tosize)])

    fromsize = 10  # Number of neurons in the second layer
    tosize = 10  # Number of neurons in the third layer
    weights2 = np.array([[random.uniform(low, high) for _ in range(fromsize)] for _ in range(tosize)])

    # Add layers to the network
    network.add_layer(4, weights1)  # First layer with weights connecting to the second layer
    network.add_layer(10, weights2)  # Second layer with weights connecting to the third layer
    network.add_layer(10)  # Third layer (output layer) with no outgoing weights

    # Draw the network
    network.draw()
