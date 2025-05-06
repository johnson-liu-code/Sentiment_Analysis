




def GloVe_train_word_vectors(
        data_file_name="data/project_data/raw_data/trimmed_training_data.csv",
        comments_limit=10,
        window_size=64,
        word_vector_length=8,
        save_data=False,
        x_max = 100,
        alpha = 0.75,
        iter = 1,
        eta = 0.1
    ):

    import pandas as pd
    import numpy as np

    import helper_functions.cooccurrence_matrix
    import helper_functions.cooccurrence_probability
    import helper_functions.word_vectors
    import machine_learning.gradient_descent

    data = pd.read_csv(data_file_name)[:comments_limit]

    text = data['comment'].values

    unique_words, cooccurrence_matrix = (
        helper_functions.cooccurrence_matrix.create_cooccurrence_matrix(
            text,
            window_size
        )
    )

    cooccurence_matrix_dataframe = (
        pd.DataFrame(
            cooccurrence_matrix,
            index=unique_words,
            columns=unique_words
        )
    )

    cooccurrence_matrix_dict = cooccurence_matrix_dataframe.to_dict()

    totals, probabilities = (
        helper_functions.cooccurrence_probability.cooccurrence_probability(
            cooccurrence_matrix_dict
        )
    )

    probabilities = pd.DataFrame.from_dict(
        probabilities,
        orient='index'
    )

    word_vectors = (
        helper_functions.word_vectors.create_word_vectors( 
            unique_words,
            word_vector_length
        )
    )

    J_over_time, word_vectors_over_time = machine_learning.gradient_descent.descent(
        unique_words,
        word_vectors,
        word_vector_length,
        probabilities,
        x_max,
        alpha,
        eta,
        iter
    )

    if save_data:
        # Save J_over_time to a binary file.
        J_over_time_save_file = 'data/project_data/training_data/test/project_J_over_time_01.npy'
        np.save(
            J_over_time_save_file,
            J_over_time
        )

        # Save word_vectors_over_time to a binary file.
        word_vectors_over_time_save_file = 'data/project_data/training_data/test/project_word_vectors_over_time_01.npy'
        np.save(
            word_vectors_over_time_save_file,
            word_vectors_over_time
        )


    return word_vectors_over_time
