



def custom_nn(word_vectors, comments):
    '''
    Custom neural network function that initializes a Keras Sequential model.
    The model is designed to work with word vectors.
    
    Args:
        1. word_vectors (dict): Pre-trained word vectors.
    
    The Function:
        - Initializes a Keras Sequential model.

    Returns:
        To terminal:
            - Prints the summary of the model.
        To file:
            - None
    '''

    import numpy as np
    import keras

    words_in_comments = [ comment.split() for comment in comments ]

    words_in_comments = [
        [word.strip('.,!?()[]{}"\'').lower() for word in comment]
        for comment in words_in_comments
    ]

    vectors_in_comments = [
        [ trained_word_vectors[word] for word in comment]
        for comment in words_in_comments
    ]


    frechet_mean_for_each_comment = [ np.mean(comment) for comment in vectors_in_comments ]
    # Convert the list of input features into a NumPy array
    X = np.array(frechet_mean_for_each_comment)


    # Example dummy target data (binary classification: 0 or 1)
    # Replace this with your actual labels
    labels = np.random.randint(0, 2, size=(len(frechet_mean_for_each_comment),))


    # Define and compile the model
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(X.shape[0],)),
            keras.layers.Dense(512, activation="relu", name='dense_0'),
            keras.layers.Dense(256, activation="relu", name='dense_1'),
            keras.layers.Dense(1, activation="sigmoid", name='output')  # For binary classification
        ]
    )

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    ##########################################################################################
    ### This doesn't work as of right now ... 
    ### Look into this further ...
    ##########################################################################################

    # Train the model
    model.fit(X, labels, epochs=10, batch_size=32, validation_split=0.2)

    all_weights = model.get_weights()

    # Example: Get weights and biases from the first Dense layer
    dense_0_weights, dense_0_biases = model.get_layer('dense_0').get_weights()

    print("Weights shape:", dense_0_weights.shape)
    print("Biases shape:", dense_0_biases.shape)

    # Save model weights
    model.save_weights('my_model_weights.h5')

    # Load weights into a model with the same architecture
    model.load_weights('my_model_weights.h5')





# Example usage of the custom_nn function.
if __name__ == "__main__":
    import numpy as np

    with open('testing_scrap_misc/scrap_data_02/word_vectors_over_time.npy', 'rb') as f:
        trained_word_vectors = np.load(f, allow_pickle=True)[-1]

    with open('data/sentences.txt', 'r') as f:
        comments = [ comment.strip() for comment in f.readlines() ]

    custom_nn(trained_word_vectors, comments)