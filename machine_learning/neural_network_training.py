



def custom_nn(
        X:list,
        labels:list
    ):


    import os

    import numpy as np

    import keras
    from keras import Model, Input
    from keras import layers


    # Define the model sequentially.
    # model = keras.Sequential(
    #     [
    #         keras.layers.Input(shape=(X.shape[1],)),
    #         keras.layers.Dense(512, activation="relu", name='dense_0'),
    #         # keras.layers.Dense(256, activation="relu", name='dense_1'),
    #         keras.layers.Dense(1, activation="sigmoid", name='output')  # For binary classification
    #     ]
    # )

    # Define the model using function API.
    # input1 = Input(shape=(X.shape[1],))
    # layer1 = layers.Dense(4)(input1)
    # layer2 = layers.concatenate([layer1, input1])
    # output1 = layers.Dense(1)(layer2)

    input1 = Input(shape=(X.shape[1],))
    layer1 = layers.Dense(8)(input1)
    layer2 = layers.Dense(8)(layer1)
    layer3 = layers.Dense(8)(layer2)
    output1 = layers.Dense(1)(layer3)


    model = Model(inputs=input1, outputs=[output1])
    model.summary()

    # Create figure of neural network.
    keras.utils.plot_model(model, show_shapes=True)

    # # Compile the model.
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    checkpoint_dir = 'data/testing_data/nn_weights_01/'
    checkpoint_filepath = os.path.join(checkpoint_dir, 'weights_{epoch:02d}.weights.h5')

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        save_freq='epoch'
    )

    # Train the model.
    model.fit(
        X,
        labels,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        callbacks=[model_checkpoint_callback],
    )

    model.save("my_model.keras")



# Example usage of the custom_nn function.
if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    import helper_functions.frechet_mean

    # Get the trained word vectors from the file.
    with open('testing_scrap_misc/scrap_data_02/word_vectors_over_time.npy', 'rb') as f:
        trained_word_vectors = np.load(f, allow_pickle=True)[-1]


    data_file_name = 'data/project_data/raw_data/trimmed_training_data.csv'
    comments_limit = 10
    # Get the comments for which we want to train the neural network on.
    data = pd.read_csv(data_file_name)[:comments_limit]['comments']

    # print(data)

    # Split each comment into their component words.
    words_in_comments = [ comment.split() for comment in data ]

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

    # # Compute the Frechet mean for each comment.
    # # The Frechet mean in our context is just the mean of all of the word vectors in a comment.
    # frechet_mean_for_each_comment = [ np.mean(comment, axis=0) for comment in vectors_in_comments ]

    # # Convert the list structure to an array.
    # X = np.array(frechet_mean_for_each_comment)

    X = helper_functions.frechet_mean.frechet_mean(vectors_in_comments)

    labels = np.random.randint(0, 2, size=(X.shape[0],))

    # Train the neural network.
    custom_nn(trained_word_vectors, X, labels)