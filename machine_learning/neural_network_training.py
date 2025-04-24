



def custom_nn(trained_word_vectors, X, labels):


    import numpy as np
    # import keras

    # print(labels)

    # # Define the model sequential.
    # model = keras.Sequential(
    #     [
    #         keras.layers.Input(shape=(X.shape[1],)),
    #         keras.layers.Dense(512, activation="relu", name='dense_0'),
    #         # keras.layers.Dense(256, activation="relu", name='dense_1'),
    #         keras.layers.Dense(1, activation="sigmoid", name='output')  # For binary classification
    #     ]
    # )

    # # Compile the model.
    # model.compile(
    #     optimizer='adam',
    #     loss='binary_crossentropy',
    #     metrics=['accuracy']
    # )

    # # Print out details of the model says.
    # model.summary()

    # # Train the model.
    # model.fit(X, labels, epochs=100, batch_size=32, validation_split=0.2)




# Example usage of the custom_nn function.
if __name__ == "__main__":
    import numpy as np

    # Get the trained word vectors from the file.
    with open('testing_scrap_misc/scrap_data_02/word_vectors_over_time.npy', 'rb') as f:
        trained_word_vectors = np.load(f, allow_pickle=True)[-1]

    # Get the comments for which we want to train the neural network on.
    with open('data/testing_data/sentences.txt', 'r') as f:
        comments = [ comment.strip() for comment in f.readlines() ]

    # Split each comment into their component words.
    words_in_comments = [ comment.split() for comment in comments ]

    # Throw away any punctuation that are attached to the words.
    words_in_comments = [
        [ word.strip('.,!?()[]{}"\'').lower() for word in comment]
        for comment in words_in_comments
    ]

    # Retrieve the vector representation of each word in each comment.
    vectors_in_comments = [
        [ trained_word_vectors[word] for word in comment]
        for comment in words_in_comments
    ]

    # Compute the Frechet mean for each comment.
    # The Frechet mean in our context is just the mean of all of the word vectors in a comment.
    frechet_mean_for_each_comment = [ np.mean(comment, axis=0) for comment in vectors_in_comments ]

    # Convert the list structure to an array.
    X = np.array(frechet_mean_for_each_comment)

    labels = np.random.randint(0, 2, size=(X.shape[0],))

    # Train the neural network.
    custom_nn(trained_word_vectors, X, labels)