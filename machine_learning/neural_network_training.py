

import numpy as np
# import keras


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
    words_in_comments = [ comment.split() for comment in comments ]

    words_in_comments = [
        [word.strip('.,!?()[]{}"\'').lower() for word in comment]
        for comment in words_in_comments
    ]

    vectors_in_comments = [
        [ trained_word_vectors[word] for word in comment]
        for comment in words_in_comments
    ]

    frechet_mean_vectors = [
        np.mean(np.array(vectors), axis=0) if len(vectors) > 0 else np.zeros(trained_word_vectors.shape[1])
        for vectors in vectors_in_comments
    ]

    print(len(frechet_mean_vectors[0]))
    print(len(frechet_mean_vectors[1]))
    print(len(frechet_mean_vectors[2]))

    # model = keras.Sequential(
    #     [
    #         # keras.layers.Input(shape=(word_vectors.shape[1],)),
    #         keras.layers.Dense(512, activation="relu"),
    #         keras.layers.Dense(256, activation="relu"),
    #     ]
    # )
    # model.summary()


# Example usage of the custom_nn function.
if __name__ == "__main__":


    with open('testing_scrap_misc/scrap_data_02/word_vectors_over_time.npy', 'rb') as f:
        trained_word_vectors = np.load(f, allow_pickle=True)[-1]

    with open('data/sentences.txt', 'r') as f:
        comments = [ comment.strip() for comment in f.readlines() ]

    custom_nn(trained_word_vectors, comments)