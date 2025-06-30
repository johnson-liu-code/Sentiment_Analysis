

if __name__ == "__main__":
    ############################################################
    # For testing purposes.
    import random
    random.seed(1994)
    ############################################################


    ############################################################
    import os
    # import argparse
    ############################################################
    import numpy as np
    import pandas as pd
    ############################################################
    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer
    ############################################################
    import functions.helper_functions.data_preprocessing
    ############################################################
    import functions.machine_learning.glove_vector_training
    import functions.machine_learning.LogBilinearModel
    ############################################################
    import functions.comment_representation.tf_idf_vectorization
    ############################################################
    import functions.machine_learning.feedforward_neural_network
    ############################################################
    import functions.data_visualization.draw_neural_network
    ############################################################



    """
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

    # args = parser.parse_args()

    # data_file_name = args.input_file_name
    # word_vectors_over_time_save_file = args.output_file_name
    # load_file_name = args.load_file_name
    # run_train_word_vectors = args.train_glove
    # save_data = args.save_glove_training_data
    # train_nn = args.train_nn
    # save_weights = args.save_weights
    """

    # part = 'preprocess_data'
    # part = 'train_word_vectors'
    # part = 'vectorize_comments'
    part = 'train_neural_network'


    # J_over_time_save_file = 'testing_scrap_misc/scrap_02/J_over_time.npy'
    # word_vectors_over_time_save_file = 'testing_scrap_misc/scrap_02/word_vectors_over_time.npy'

    if part == 'preprocess_data':
        # unique_words, cooccurrence_matrix, probabilities, J_over_time, word_vectors_over_time = functions.machine_learning.glove_vector_training.GloVe_train_word_vectors(
        #     data_file_name="data/project_data/raw_data/trimmed_training_data.csv",
        #     comments_limit=100,
        #     window_size=10,
        #     word_vector_length=100,
        #     x_max = 100,
        #     alpha = 0.75,
        #     iter = 1,
        #     eta = 0.01,
        #     save_dir="testing_scrap_misc/scrap_02"
        # )

        
        unique_words, cooccurrence_matrix, probabilities, text, labels = (
            functions.helper_functions.data_preprocessing.data_preprocessing(
                data_file_name="data/project_data/raw_data/trimmed_training_data.csv",
                comments_limit=1000,
                window_size=10,
            )
        )

        save_dir = 'testing_scrap_misc/scrap_02/'
        unique_words_save_file = save_dir + 'unique_words.npy'
        cooccurrence_matrix_save_file = save_dir + 'cooccurrence_matrix.npy'
        probabilities_save_file = save_dir + 'cooccurrence_probability_matrix.npy'
        text_save_file = save_dir + 'text.npy'
        labels_save_file = save_dir + 'labels.npy'

        # Save data to files.
        print(f'Save preprocessed data to files in {save_dir}...')
        np.save(
            unique_words_save_file,
            unique_words
        )
        
        np.save(
            cooccurrence_matrix_save_file,
            cooccurrence_matrix
        )

        # Save data to files.

        np.save(
            unique_words_save_file,
            unique_words
        )
        
        np.save(
            cooccurence_matrix_save_file,
            cooccurrence_matrix
        )

        np.save(
            probabilities_save_file,
            probabilities
        )

        np.save(
            J_over_time_save_file,
            J_over_time
        )

        # word_vectors_over_time is a list of list of word vectors.
        np.save(
            probabilities_save_file,
            probabilities
        )

        np.save(
            text_save_file,
            text
        )

        np.save(
            labels_save_file,
            labels
        )
        # np.save(
        #     J_over_time_save_file,
        #     J_over_time
        # )

        # word_vectors_over_time is a list of list of word vectors.
        # np.save(
        #     word_vectors_over_time_save_file,
        #     word_vectors_over_time
        # )

    elif part == 'train_word_vectors':

        save_dir = 'testing_scrap_misc/scrap_02/'
        unique_words_save_file = save_dir + 'unique_words.npy'
        cooccurrence_matrix_save_file = save_dir + 'cooccurrence_matrix.npy'
        probabilities_save_file = save_dir + 'cooccurrence_probability_matrix.npy'
        text_save_file = save_dir + 'text.npy'

        # Load the preprocessed data.
        print(f'Loading preprocessed data from files in {save_dir}...')
        unique_words = np.load(unique_words_save_file, allow_pickle=True)
        cooccurrence_matrix = np.load(cooccurrence_matrix_save_file, allow_pickle=True)
        probabilities = np.load(probabilities_save_file, allow_pickle=True)

        # Convert the cooccurrence matrix to a torch tensor.
        cooccurrence_probability_tensor = torch.tensor(probabilities)
        cooccurrence_probability_tensor = cooccurrence_probability_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Train the word vectors using PyTorch.
        print("Training word vectors using PyTorch...")
        word_vectors_over_time = functions.machine_learning.LogBilinearModel.train(
            cooc_matrix=cooccurrence_probability_tensor,
            embedding_dim=200,
            epochs=1,
            batch_size=256,
            learning_rate=0.01,
            x_max=100,
            alpha=0.75,
            num_workers=4,
            save_dir=save_dir + 'training_logs',
            use_gpu=True
        )


    elif part == 'vectorize_comments':
        # data_file_name="data/project_data/raw_data/trimmed_training_data.csv"
        # comments_limit = 1000
        # print(f"Reading data from {data_file_name} with a limit of {comments_limit} comments...")
        # data = pd.read_csv(data_file_name).sample(n=comments_limit, random_state=94)
        # data = pd.read_csv(data_file_name)[:comments_limit]
        # data = data.dropna(subset=['comment'])
        # Collect only the text from the data.
        # np.ndarray
        # text = data['comment'].values
        text_save_file = 'testing_scrap_misc/scrap_02/text.npy'
        text = np.load(text_save_file, allow_pickle=True)

        # Print indices and values where text is not a string
        non_string_indices = [(i, t) for i, t in enumerate(text) if not isinstance(t, str)]
        if non_string_indices:
            print("Non-string values found in 'text' at the following indices:")
            for idx, val in non_string_indices:
                print(f"Index {idx}: {val} (type: {type(val)})")
        else:
            print("All values in 'text' are strings.")

        # print(text[:3])
        # print(type(text))
        # print(text.shape)

        # ---------------------------
        # Step 1: Fit a TF-IDF Vectorizer
        # ---------------------------
        # This calculates both TF and IDF values across the corpus
        print('Initializing tf-idf vectorizer...')
        vectorizer = TfidfVectorizer(lowercase=True, tokenizer=str.split, token_pattern=None)
        vectorizer.fit(text)

        # ---------------------------
        # Step 2: Build Vocabulary and Fake Word Embeddings
        # ---------------------------
        # Vocabulary: word → index
        print('Indexing unqiue words...')
        word_to_idx = {word: idx for idx, word in enumerate(vectorizer.get_feature_names_out())}

        # print('word_to_idx:')
        # print(word_to_idx)

        # Fake embeddings for demonstration (dim=50)
        # embedding_dim = 50
        # embedding_matrix = torch.randn(len(word_to_idx), embedding_dim)
        # print(embedding_matrix)

        print('Loading the trained word vectors...')
        trained_word_vectors_file = 'testing_scrap_misc/scrap_02/training_logs/final_word_vectors.pt'
        word_vectors_matrix = torch.load(trained_word_vectors_file)
        # print(word_vectors)

        # comment = text[1]
        # print('Aggregating word vectors in comment using TF-IDF weighting...')
        # sentence_vector = functions.comment_representation.tf_idf.embed_comment_tfidf(comment, vectorizer, word_to_idx, word_vectors_matrix)
        # print('comment:')
        # print(comment)
        # print('sentence_vector:')
        # print(sentence_vector)
        # print(sentence_vector.shape)

        print('Saving the vectorized comments...')
        output_file_name = 'testing_scrap_misc/scrap_02/vectorized_comments.npy'
        functions.comment_representation.tf_idf_vectorization.vectorize_comments_with_tfidf(text, vectorizer, word_vectors_matrix, output_file_name)



    elif part == 'train_neural_network':
        print('Loading vectorized comments and corresponding labels...')
        vectorized_comments_file_name = 'testing_scrap_misc/scrap_02/vectorized_comments.npy'
        vectorized_comments = np.load(vectorized_comments_file_name)
        # print(vectorized_comments.shape)
        labels = np.load('testing_scrap_misc/scrap_02/labels.npy')
        # print(labels.shape[0])

        print('Training the feedforward neural network...')
        functions.machine_learning.feedforward_neural_network.custom_fnn(vectorized_comments, labels)

        

    elif part == 'use_trained_model':
        pass

    else:
        pass


    # Load trained word vectors.
        # if run_train_word_vectors:

    # else:
    #     word_vectors_over_time = np.load(
    #         load_file_name,
    #         allow_pickle=True
    #     )
    ###############



    '''
    # Extract the last list of word vectors.
    trained_word_vectors = word_vectors_over_time[-1]

    # Train the neural network.
    if train_nn:
        ############################################################
        import keras
        ############################################################

        # Number of data points.
        comments_limit = 64
        data = pd.read_csv(data_file_name)
        # Randomly collect data points.
        data = random.sample(data, comments_limit)
        
        # Extract only the actual comments and their labels ( either sarcastic or not sarcastic ).
        # Each comment is a single string.
        comments = data['comment'].values
        labels = data['label'].values

        # Split each comment into a list of strings and store all of these lists into a list.
        # comments = [ [ *comment01* ], [ *comment02* ], ... ]
        # [ *comment01* ] = [ *substring01*, *substring02*, *substring03*, ... ]
        comments = [ comment.split() for comment in comments ]

        # Remove words in each comment that does not appear in the trained_word_vectors dictionary.
        # 20250610 note: Why is this neccessary? Elaborate.
        comments = [
            [ word for word in comment if word in trained_word_vectors
            ] for comment in comments
        ]

        # Each word in each comment is represented by an embedding vector.
        vectorized_comments = [
            [ trained_word_vectors[word] for word in comment
            ] for comment in comments
        ]

        # Taking a central measure of all of the word vectores in a comment so that each
        # data point ( comment ) is represented by a single vector.
        centered_comments = functions.helper_functions.frechet_mean.frechet_mean(
            vectorized_comments,
            word_vector_length = 8
        )

        # Train the neural network on the comments.
        functions.machine_learning.neural_network_training.custom_nn(
            centered_comments,
            labels
        )
    
    if save_weights:
        import keras

        reconstructed_model = keras.models.load_model("my_model.keras")

        checkpoint_dir = 'data/testing_data/nn_weights_01/'

        weights_over_time = []

        epochs = len([name for name in os.listdir('.') if os.path.isfile(checkpoint_dir)])

        for epoch in epochs:
            # Load weights for the current epoch.
            weights_path = os.path.join(checkpoint_dir, f'weights_{epoch:02d}.weights.h5')
            reconstructed_model.load_weights(weights_path)
            weights = reconstructed_model.get_weights()
            weights_over_time.append(weights)

        # Save the weights to a file.
        weights_dict = {f'epoch_{i+1:04}': weights for i, weights in enumerate(weights_over_time)}
        np.save('weights_over_time.npy', weights_dict)

    saved_weights_over_time = np.load('weights_over_time.npy', allow_pickle=True) 
    epoch_keys = saved_weights_over_time.item().keys()

    # Collect the weights for each epoch into a list.
    weights_over_time = [saved_weights_over_time.item()[key] for key in epoch_keys]

    frames = len(weights_over_time)
    
    nn_viz = functions.data_visualization.draw_neural_network.NeuralNetworkVisualizer()
    nn_viz.add_layer(8, "Input Layer")
    nn_viz.add_layer(8, "Hidden Layer 1 (Dense)")
    nn_viz.add_layer(8, "Hidden Layer 2 (Dense)")
    nn_viz.add_layer(8, "Hidden Layer 3 (Dense)")
    nn_viz.add_layer(1, "Output Layer")

    initial_weights = weights_over_time[0]

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


'''

######################################################################################################
# Notes
#######
# 1. Three things to try:
#   a. FNN
#      Feedforward Neural Network using aggregated word vectors for each comment.
            # Pros:
                # Simple and fast to train
                # Works well with aggregated embeddings (e.g., TF-IDF-weighted GloVe vectors)
                # Fewer parameters → less overfitting on small datasets
            # Cons:
                # Ignores word order and syntax
                # Cannot model phrases like “I just love waiting in line” where sarcasm depends on context
                # Performance plateaus if the model lacks sequential information
            # Use FNNs if:
                # You're using averaged or TF-IDF-weighted embeddings
                # You want a fast, simple baseline
#   b. CNN
#      Convolutional Neural Networks using stacks of word vectors with padding for shorter comments.
            # Pros:
                # Captures local n-gram patterns that are useful for sarcasm (e.g., “great job”, “love that”)
                # Faster to train than RNNs
                # Some resistance to word order noise
            # Cons:
                # Limited to local context (can’t model long-range dependencies)
                # Can miss sarcasm that builds over multiple clauses
            # Use CNNs if:
                # You have access to sequence-preserving embeddings (e.g., [sequence_length, embedding_dim])
                # Sarcasm is often expressed in short phrases
#   c. RNN
#      Recurrent Neural Networks feeding word vectors in sequentially.
            # Pros:
                # Models word order and long-range dependencies
                # Good for subtle sarcasm that builds over the sentence
                # LSTM/GRU handles negations, sentiment shifts, and intensifiers (e.g., “Oh yeah, that’s exactly what I wanted…”)
            # Cons:
                # Slower training than CNNs/FNNs
                # More prone to overfitting, especially with small data
                # Vanilla RNNs can struggle with long sequences (use LSTM or GRU)
            # Use RNNs if:
                # Sarcasm depends on word order or context buildup
                # You’re okay with slightly longer training times
# 
# 1. Revise the structure/logic of the argument handling in the beginning parts of main.py.
#    Make it so that the user can run each part of the code sequentially.
#
# 1. Go through __name__ == "__main__" in each file to make sure the code still works / is up-to-date.
#
# ------------------------------------------------------------------- #
# ---------------------- Possible Improvements ---------------------- #
# ------------------------------------------------------------------- #
# 1. CNNs over Word Embeddings
#       - Cannot squash comments into uniform comment vectors. Need full comment and padding.
#       - Apply 1D convolution filters to detect n-gram patterns (e.g., sarcasm markers).
#       - Captures local word patterns
# 2. Recurrent Neural Networks (RNNs / LSTMs / GRUs)
#       Feed word embeddings sequentially into an RNN to produce a context-sensitive representation.
#       Captures word order
#       Maintains directional context
#       Final hidden state or an attention-weighted sum can represent the whole comment

