


def GloVe_train_word_vectors(
        data_file_name="data/project_data/raw_data/trimmed_training_data.csv",
        comments_limit=10,
        window_size=64,
        word_vector_length=8,
        x_max = 100,
        alpha = 0.75,
        iter = 1,
        eta = 0.1
    ):
    """
    _summary_

    Args:
        1. data_file_name (str, optional):
            Name of input file with the training data.
            Defaults to "data/project_data/raw_data/trimmed_training_data.csv".

        2. comments_limit (int, optional):
            The number of comments to use to collect words from.
            Defaults to 10.

        3. window_size (int, optional):
            Window size for collecting context words.
            Defaults to 64.

        4. word_vector_length (int, optional):
            Size of the vector that represents words.
            Defaults to 8.

        5. x_max (int, optional):
            Constant used to compute the scaling factor for the g value used in the gradient descent portion of the GloVe model.
            Defaults to 100.

        6. alpha (float, optional):
            Used for scaling factor for the g value in gradient descent.
            Defaults to 0.75.

        7. iter (int, optional):
            Number of gradient descent iterations.
            Defaults to 1.

        8. eta (float, optional):
            Learning rate for gradient descent.
            Defaults to 0.1.

    The Function:
        _description_

    Returns:
        _type_: _description_
    """

    ###########################################################################
    import pandas as pd
    import numpy as np
    ###########################################################################
    import functions.helper_functions.cooccurrence_matrix
    import functions.helper_functions.cooccurrence_probability
    import functions.helper_functions.word_vectors
    import functions.machine_learning.gradient_descent
    ###########################################################################

    # Read the data into a DataFrame.
    data = pd.read_csv(data_file_name)[:comments_limit]

    # Collect only the text from the data.
    # All of the comments are concatenated into a single string of text.
    text = data['comment'].values

    # Generate a list of unique words from the text.
    # Generate a co-occurrence matrix from the unique words by scanning through the comments.
    unique_words, cooccurrence_matrix = (
        functions.helper_functions.cooccurrence_matrix.create_cooccurrence_matrix(
            text,
            window_size
        )
    )

    # Generate a co-occurence matrix DataFrame.
    cooccurence_matrix_dataframe = (
        pd.DataFrame(
            cooccurrence_matrix,
            index=unique_words,
            columns=unique_words
        )
    )

    # Generate a co-occurrence matrix dictionary where each unique word is a key and the
    # corresponding value is list of co-occurrences with all of the other words.
    cooccurrence_matrix_dict = cooccurence_matrix_dataframe.to_dict()

    # Convert co-occurrence frequencies into probabilities.
    totals, probabilities = (
        functions.helper_functions.cooccurrence_probability.cooccurrence_probability(
            cooccurrence_matrix_dict
        )
    )

    probabilities = pd.DataFrame.from_dict(
        probabilities,
        orient='index'
    )

    # Initialize word vectors for each unique word.
    word_vectors = (
        functions.helper_functions.word_vectors.create_word_vectors( 
            unique_words,
            word_vector_length
        )
    )

    J_over_time, word_vectors_over_time = functions.machine_learning.gradient_descent.descent(
        unique_words,
        word_vectors,
        word_vector_length,
        probabilities,
        x_max,
        alpha,
        eta,
        iter
    )

    ######################
    # Phase this out. Data saving should be done in main.py.
    # if save_data:
    #     # Save J_over_time to a binary file.
    #     J_over_time_save_file = 'data/project_data/training_data/test/project_J_over_time_01.npy'
    #     np.save(
    #         J_over_time_save_file,
    #         J_over_time
    #     )

    #     # Save word_vectors_over_time to a binary file.
    #     word_vectors_over_time_save_file = 'data/project_data/training_data/test/project_word_vectors_over_time_01.npy'
    #     np.save(
    #         word_vectors_over_time_save_file,
    #         word_vectors_over_time
    #     )
    ######################

    return unique_words, cooccurrence_matrix_dict, probabilities, J_over_time, word_vectors_over_time
