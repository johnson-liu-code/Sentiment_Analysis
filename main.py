def train_nn_on_GloVe_vectors(trained_word_vectors):
    import machine_learning.neural_network_training

    


if __name__ == "__main__":
    import os
    import argparse

    import numpy as np
    import pandas as pd
    
    import helper_functions.frechet_mean

    import machine_learning.glove_vector_training
    import machine_learning.neural_network_training

    import data_visualization.draw_neural_network



    parser_description = "This script contains the code for running the word vector training as well as running the neural network."
    parser = argparse.ArgumentParser(description=parser_description)

    # Parse the command line arguments to tell the script whether to run the GloVe training or to pull data from .npy files that already exist.
    parser.add_argument(
        "--input_file_name",
        default="data/project_data/raw_data/trimmed_training_data.csv",
        help="Path to data used for training.",
    )
    parser.add_argument(
        "--output_file_name",
        default="data/project_data/training_data/test/project_word_vectors_over_time_01.npy",
        help="Path to save word vector training data.",
    )
    parser.add_argument(
        "--load_file_name",
        default="test_trained_word_vectors.npy",
        help="Path to load saved word vector training data.",
    )
    parser.add_argument(
        "--train_glove",
        action="store_true",
        default=False,
        help="Run the GloVe training.",
    )
    parser.add_argument(
        "--save_glove_training_data",
        action="store_true",
        default=False,
        help="Save the trained GloVe vectors to file.",
    )
    parser.add_argument(
        "--train_nn",
        action="store_true",
        default=False,
        help="Train the neural network.",
    )
    parser.add_argument(
        "--save_weights",
        action="store_true",
        default=False,
        help="Save the weights in each epoch to a single file as a dictionary (key: epoch, value: weights)."
    )

    args = parser.parse_args()

    data_file_name = args.input_file_name
    word_vectors_over_time_save_file = args.output_file_name
    load_file_name = args.load_file_name
    run_train_word_vectors = args.train_glove
    save_data = args.save_glove_training_data
    train_nn = args.train_nn
    save_weights = args.save_weights

    if run_train_word_vectors:
        word_vectors_over_time = machine_learning.glove_vector_training.GloVe_train_word_vectors(
            data_file_name="data/project_data/raw_data/trimmed_training_data.csv",
            comments_limit=10,
            window_size=64,
            word_vector_length=8,
            save_data=False,
            x_max = 100,
            alpha = 0.75,
            iter = 100,
            eta = 0.1
        )

        np.save(
            word_vectors_over_time_save_file,
            word_vectors_over_time
        )

    else:
        word_vectors_over_time = np.load(
            load_file_name,
            allow_pickle=True
        )

    trained_word_vectors = word_vectors_over_time[-1]

    if train_nn:
        import keras

        comments_limit = 64
        data = pd.read_csv(data_file_name)[:comments_limit]
        comments = data['comment'].values
        labels = data['label'].values

        comments = [ comment.split() for comment in comments ]

        # Remove words in each comment that does not appear in the trained_word_vectors dictionary
        comments = [
            [ word for word in comment if word in trained_word_vectors
            ] for comment in comments
        ]

        # print(f'Comments: {comments}')

        # vectorized_comments = [
        #     [ trained_word_vectors.get(
        #             word,
        #             np.zeros_like( next( iter( trained_word_vectors.values() ) ) )
        #         ) for word in comment
        #     ] for comment in comments
        # ]

        vectorized_comments = [
            [ trained_word_vectors[word] for word in comment
            ] for comment in comments
        ]

        # print(f'Vectorized comments: {vectorized_comments}')

        # print(f'Word vector keys:\n{trained_word_vectors.keys()}\n')
        # print(f'Trained word vectors: {trained_word_vectors}')
    
        # i = 1
        # print(f'Comment {i}: {comments[i]}')
        # print(f'Vectorized comment {i}: {vectorized_comments[i]}')

        centered_comments = helper_functions.frechet_mean.frechet_mean(
            vectorized_comments,
            word_vector_length = 8
        )
        # print(f"Frech'ed comments: {centered_comments}")
        # print(f'Labels: {labels}')
        # print(len(centered_comments))
        # print(len(labels))

        machine_learning.neural_network_training.custom_nn(
            centered_comments,
            labels
        )
    
    if save_weights:
        import keras

        reconstructed_model = keras.models.load_model("my_model.keras")

        checkpoint_dir = 'data/testing_data/nn_weights_01/'

        weights_over_time = []

        epochs = np.arange(1,11)
        # epochs = len([name for name in os.listdir('.') if os.path.isfile(checkpoint_dir)])

        for epoch in epochs:
            # Load weights for the current epoch
            weights_path = os.path.join(checkpoint_dir, f'weights_{epoch:02d}.weights.h5')
            reconstructed_model.load_weights(weights_path)
            weights = reconstructed_model.get_weights()
            # print(weights)
            weights_over_time.append(weights)

        # Save the weights to a file.
        weights_dict = {f'epoch_{i+1:04}': weights for i, weights in enumerate(weights_over_time)}
        # print(weights_dict)
        # np.savez('weights_over_time.npz', **weights_dict)
        np.save('weights_over_time.npy', weights_dict)

    saved_weights_over_time = np.load('weights_over_time.npy', allow_pickle=True) 
    epoch_keys = saved_weights_over_time.item().keys()

    # Collect the weights for each epoch into a list.
    weights_over_time = [saved_weights_over_time.item()[key] for key in epoch_keys]
    # for weight in weights_over_time:
        # print(weight)
        # print(f'weights[{i}].shape: {weight.shape}')

    # print(weights_over_time[0])

    frames = len(weights_over_time)
    
    nn_viz = data_visualization.draw_neural_network.NeuralNetworkVisualizer()
    nn_viz.add_layer(8, "Input Layer")
    nn_viz.add_layer(8, "Hidden Layer 1 (Dense)")
    nn_viz.add_layer(8, "Hidden Layer 2 (Dense)")
    nn_viz.add_layer(8, "Hidden Layer 3 (Dense)")
    nn_viz.add_layer(1, "Output Layer")

    initial_weights = weights_over_time[0]
    
    # for layer_weight in initial_weights:
    #     print(layer_weight)
    #     print('\n')

    # print(initial_weights[0])
    # print(initial_weights[1])
    # print(initial_weights[2])
    # print(initial_weights[3])

    nn_viz.initialize_weights(initial_weights)

    nn_viz.draw(neuron_spacing=0.06, animate=True, save_figure=True, frames=frames, weight_frames=weights_over_time)
    
    nn_viz.draw(neuron_spacing=0.06, animate=False, save_figure=True, frames=frames, weight_frames=weights_over_time)


    # Plot the weights for each layer over time.
    weights_difference_over_time = []
    for i in range(len(weights_over_time)-1):
        weights_difference = []
        for j in range(len(weights_over_time[i])):
            weights_difference.append(weights_over_time[i+1][j] - weights_over_time[i][j])
        weights_difference_over_time.append(weights_difference)

    # print(weights_difference_over_time)
    # Plot the difference in weights for the hidden layers over time.

    first_layer_weights_difference_over_time = [weights_difference[0] for weights_difference in weights_difference_over_time]

    # Plot the difference in weights for the first hidden layer over time as a heatmap.
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract the weight differences for the first hidden layer over time.
    first_hidden_layer_weights_diff = [
        weights_difference[1] for weights_difference in weights_difference_over_time
    ]

    # Convert the list of weight differences into a 2D array for plotting.
    # Each row corresponds to an epoch, and each column corresponds to a weight.
    heatmap_data = np.array([weights.flatten() for weights in first_hidden_layer_weights_diff])

    # Take the log of the absolute value of the differences.
    log_heatmap_data = np.log(np.abs(heatmap_data) + 1e-8)  # Add a small value to avoid log(0).

    # Plot the heatmap.
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(log_heatmap_data, aspect='auto', cmap='viridis', interpolation='nearest')

    # Add colorbar for reference.
    fig.colorbar(cax, ax=ax)

    # Add labels and title.
    ax.set_title("Log of Weight Differences Over Time (First Hidden Layer)")
    ax.set_xlabel("Weight Index")
    ax.set_ylabel("Epoch")

    # Show the plot.
    plt.show()


    print(first_hidden_layer_weights_diff[0])

    # Animate the change in weights for the first layer over time.
    import matplotlib.animation as animation

    # Extract the weight differences for the first layer over time.
    first_layer_weights_difference_over_time = [
        weights_difference[0] for weights_difference in weights_difference_over_time
    ]

    # Ensure the weights are reshaped into 8x8 matrices for the heatmap.
    heatmap_data = [weights.reshape(8, 8) for weights in first_layer_weights_difference_over_time]

    # Create the figure and axis for the animation.
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.imshow(heatmap_data[0], cmap='viridis', interpolation='nearest', aspect='auto')
    fig.colorbar(cax, ax=ax)

    # Add labels and title.
    ax.set_title("Change in Weights Over Time (First Layer)")
    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("Neuron Index")

    # Function to update the heatmap for each frame.
    def update(frame):
        cax.set_array(heatmap_data[frame])
        ax.set_title(f"Change in Weights Over Time (Epoch {frame + 1})")
        return cax,

    # Create the animation.
    ani = animation.FuncAnimation(
        fig, update, frames=len(heatmap_data), interval=500, blit=False
    )

    # Show the animation.
    plt.show()

    # Optionally, save the animation as a video or GIF.
    # ani.save("weights_change_animation.mp4", writer="ffmpeg")

