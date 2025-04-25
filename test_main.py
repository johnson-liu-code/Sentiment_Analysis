


 

def GloVe_train_word_vectors(
        data_file_name="data/project_data/raw_data/trimmed_training_data.csv",
        comments_limit=10,
        window_size=100,
        word_vector_length = 10
    ):

    import pandas as pd
    import numpy as np

    import helper_functions.cooccurrence_matrix
    import helper_functions.cooccurrence_probability
    import helper_functions.word_vectors
    import machine_learning.gradient_descent

    data = pd.read_csv(data_file_name)[:comments_limit]
    # print(data)

    text = data['comment'].values
    # print(text)

    unique_words, cooccurrence_matrix = (
        helper_functions.cooccurrence_matrix.create_cooccurrence_matrix(
            text, window_size
        )
    )

    # print(f'Unique words: {unique_words}\n')
    # print(f'Coocurrence_matrix: {cooccurrence_matrix}')

    # helper_functions.cooccurrence_matrix.plot_cooccurrence_heatmap(
    #     unique_words, cooccurrence_matrix, show=True)

    cooccurence_matrix_dataframe = (
        pd.DataFrame(
            cooccurrence_matrix,
            index=unique_words,
            columns=unique_words
        )
    )

    # print(cooccurence_matrix_dataframe)

    cooccurrence_matrix_dict = cooccurence_matrix_dataframe.to_dict()

    totals, probabilities = (
        helper_functions.cooccurrence_probability.cooccurrence_probability(
            cooccurrence_matrix_dict
        )
    )

    # print(f'Totals: {totals}')
    # print(f'Probabilities: {probabilities}')

    probabilities = pd.DataFrame.from_dict(probabilities, orient='index')
    # print(probabilities)

    word_vectors = (
        helper_functions.word_vectors.create_word_vectors( 
            unique_words, word_vector_length
        )
    )

    # print(word_vectors)

    x_max = 100
    alpha = 0.75
    iter = 2
    eta = 0.1

    J_over_time, word_vectors_over_time = machine_learning.gradient_descent.descent(
        unique_words, word_vectors, probabilities, x_max, alpha, eta, iter )

    # Save J_over_time to binary file
    J_over_time_save_file = 'project_J_over_time_01.npy'
    np.save(J_over_time_save_file, J_over_time)

    # Save word_vectors_over_time to binary file
    word_vectors_over_time_save_file = 'project_word_vectors_over_time_01.npy'
    np.save(word_vectors_over_time_save_file, word_vectors_over_time)


def train_nn_on_GloVe_vectors(trained_word_vectors):
    import machine_learning.neural_network_training

    
    


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import machine_learning.neural_network_training

    # GloVe_train_word_vectors(data_file_name = "data/project_data/trimmed_training_data.csv", comments_limit=10)

    word_vectors_over_time_save_file = 'data/project_data/training_data/test/project_word_vectors_over_time_01.npy'
    word_vectors_over_time = np.load(word_vectors_over_time_save_file, allow_pickle=True)

    trained_word_vectors = word_vectors_over_time[-1]

    data_file_name = "data/project_data/raw_data/trimmed_training_data.csv"
    comments_limit=10

    # Get the comments for which we want to train the neural network on.
    data = pd.read_csv(data_file_name)[:comments_limit]['comment']

    print(data)

    # # Split each comment into their component words.
    # words_in_comments = [ comment.split() for comment in comments ]

    # # Throw away any punctuation that are attached to the words.
    # words_in_comments = [
    #     [ word.strip('.,!?()[]{}"\'').lower() for word in comment]
    #     for comment in words_in_comments
    # ]

    # # Retrieve the vector representation of each word in each comment.
    # vectors_in_comments = [
    #     [ trained_word_vectors[word] for word in comment]
    #     for comment in words_in_comments
    # ]

    # X = word_vectors_over_time.neural_network_training.frechet_mean(vectors_in_comments)