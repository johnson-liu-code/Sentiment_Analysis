import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

class NeuralNetworkVisualizer:
    def __init__(self):
        self.layers = []
        self.weights = []
        self.layer_names = []

    def add_layer(self, num_neurons, layer_name=""):
        """Add a layer to the neural network."""
        self.layers.append(num_neurons)
        self.layer_names.append(layer_name)

    def initialize_weights(self, initial_weights):
        """Initialize weights from the first frame of weight_frames."""
        if len(self.layers) < 2:
            raise ValueError("At least two layers are required to initialize weights.")
        if len(initial_weights) != len(self.layers) - 1:
            raise ValueError("The number of weight matrices must match the number of layer connections.")
        for i, weight_matrix in enumerate(initial_weights):
            expected_shape = (self.layers[i], self.layers[i + 1])
            if weight_matrix.shape != expected_shape:
                raise ValueError(f"Weight matrix dimensions {weight_matrix.shape} do not match the expected dimensions {expected_shape}.")
        self.weights = initial_weights

    def draw(self, neuron_spacing=0.04, animate=False, frames=50, weight_frames=None):
        """Draw the neural network with optional animation."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('equal')
        plt.xticks([])
        plt.yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Calculate layer positions
        x_positions = np.linspace(0, 1, len(self.layers))
        y_positions = []
        for num_neurons in self.layers:
            total_height = 0.8
            start_y = 0.5 - (num_neurons - 1) * neuron_spacing / 2
            y_positions.append([start_y + i * neuron_spacing for i in range(num_neurons)])

        # Draw neurons
        neuron_radius = 0.02
        for i, (x, y) in enumerate(zip(x_positions, y_positions)):
            for neuron_y in y:
                circle = plt.Circle((x, neuron_y), neuron_radius, color='black', fill=False, lw=1.5)
                ax.add_artist(circle)
            if self.layer_names[i]:
                label_y_position = min(y) - 0.05
                ax.text(x, label_y_position, self.layer_names[i], ha='center', fontsize=10)

        # Draw weights
        lines = []
        for i, weight_matrix in enumerate(self.weights):
            for j, y1 in enumerate(y_positions[i]):
                for k, y2 in enumerate(y_positions[i + 1]):
                    weight = weight_matrix[j, k]
                    color = plt.cm.bwr((weight + 1) / 2)
                    x_start = x_positions[i] + neuron_radius
                    x_end = x_positions[i + 1] - neuron_radius
                    line, = ax.plot([x_start, x_end], [y1, y2], color=color, lw=0.5)
                    lines.append((line, i, j, k))

        # Add colorbar for weights
        sm = plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Weight Value', fontsize=10)

        if animate:
            if not weight_frames:  # Check if weight_frames is empty
                print("weight_frames is empty. Showing a still image.")
                plt.show()
                return

            if len(weight_frames) != frames:
                raise ValueError("weight_frames must be provided and match the number of frames.")

            def update(frame):
                # Update weights for animation
                current_weights = weight_frames[frame]
                for line, i, j, k in lines:
                    weight = current_weights[i][j, k]
                    color = plt.cm.bwr((weight + 1) / 2)
                    line.set_color(color)

            anim = FuncAnimation(fig, update, frames=frames, interval=200, repeat=True)
            plt.show()
        else:
            plt.show()


# Example usage
if __name__ == "__main__":
    nn_viz = NeuralNetworkVisualizer()
    
    a = 8
    b = 10
    z = 1

    nn_viz.add_layer(a, "Input Layer")
    nn_viz.add_layer(b, "Hidden Layer 1")
    nn_viz.add_layer(b, "Hidden Layer 2")
    nn_viz.add_layer(z, "Output Layer")

    frames = 100
    # Generate weight frames for animation
    weight_frames = []
    for _ in range(frames):
        frame_weights = [
            np.random.uniform(-1, 1, (a, b)),
            np.random.uniform(-1, 1, (b, b)),
            np.random.uniform(-1, 1, (b, z))
        ]
        weight_frames.append(frame_weights)

    # print(weight_frames[0])
    # print(len(weight_frames[0]))

    # Initialize weights using the first frame
    # nn_viz.initialize_weights(weight_frames[0])

    # Pass weight frames to the draw method
    # nn_viz.draw(neuron_spacing=0.06, animate=True, frames=frames, weight_frames=weight_frames)