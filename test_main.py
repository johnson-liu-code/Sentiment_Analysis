


if __name__ == "__main__":

    import pandas as pd
    import numpy as np

    import helper_functions.cooccurrence_matrix
    import helper_functions.cooccurrence_probability
    import helper_functions.word_vectors
    import machine_learning.gradient_descent

    original_data_file_name = "data/project_data/trimmed_training_data.csv"

    comments_limit = 100

    data = pd.read_csv(original_data_file_name)[:comments_limit]
    # print(data)

    text = data['comment'].values
    # print(text)

    window_size = 30

    unique_words, cooccurrence_matrix = (
        helper_functions.cooccurrence_matrix.create_cooccurrence_matrix(
            text, window_size ) )

    # print(f'Unique words: {unique_words}\n')
    # print(f'Coocurrence_matrix: {cooccurrence_matrix}')

    # helper_functions.cooccurrence_matrix.plot_cooccurrence_heatmap(
    #     unique_words, cooccurrence_matrix, show=True)

    cooccurence_matrix_dataframe = (
            pd.DataFrame(
                        cooccurrence_matrix, index = unique_words, columns = unique_words
                        )
            )

    # print(cooccurence_matrix_dataframe)

    cooccurrence_matrix_dict = cooccurence_matrix_dataframe.to_dict()

    totals, probabilities = (
        helper_functions.cooccurrence_probability.cooccurrence_probability(
            cooccurrence_matrix_dict ) )

    # print(f'Totals: {totals}')
    # print(f'Probabilities: {probabilities}')

    probabilities = pd.DataFrame.from_dict(probabilities, orient='index')
    # print(probabilities)

    word_vectors = (
        helper_functions.word_vectors.create_word_vectors( 
            unique_words, len(unique_words) ) )

    # print(word_vectors)

    x_max = 100
    alpha = 0.75
    iter = 10
    eta = 0.1

    J_over_time, word_vectors_over_time = machine_learning.gradient_descent.descent(
        unique_words, word_vectors, probabilities, x_max, alpha, eta, iter )

    # Save J_over_time to binary file
    J_over_time_save_file = 'project_J_over_time_01.npy'
    np.save(J_over_time_save_file, J_over_time)

    # Save word_vectors_over_time to binary file
    word_vectors_over_time_save_file = 'project_word_vectors_over_time_01.npy'
    np.save(word_vectors_over_time_save_file, word_vectors_over_time)