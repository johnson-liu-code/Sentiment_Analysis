
def data_preprocessing(
    data_file_name: str,
    comments_limit: int,
    window_size: int
    ):
    """
    Preprocess data for training word vectors using GloVe.

    Args:
        - data_file_name (str): Name of the input file with the training data.
        - comments_limit (int): The number of comments to use to collect words from.
        - window_size (int): Window size for collecting context words.

    Returns:
        - unique_words (list): List of unique words.
        - cooccurrence_matrix_dict (dict): Dictionary of co-occurrence matrices.
        - probabilities (np.ndarray): 2D array of co-occurrence probabilities.
    """
    
    ###########################################################################
    import pandas as pd
    import numpy as np
    ###########################################################################
    import functions.helper_functions.cooccurrence_matrix
    import functions.helper_functions.cooccurrence_probability
    ###########################################################################

    # Read the data into a DataFrame.
    # Randomly pick the data points.
    print(f"Reading data from {data_file_name} with a limit of {comments_limit} comments...")
    data = pd.read_csv(data_file_name).sample(n=comments_limit, random_state=94)
    # data = pd.read_csv(data_file_name)[:comments_limit]

    data = data.dropna(subset=['comment'])

    # Collect only the text from the data.
    # np.ndarray
    text = data['comment'].values

    # Generate a list of unique words from the text.
    # Generate a co-occurrence matrix from the unique words by scanning through the comments.
    # This returns a 2D array for the co-occurrence matrix.
    print("Generating unique words and co-occurrence matrix...")
    unique_words, cooccurrence_matrix = (
        functions.helper_functions.cooccurrence_matrix.create_cooccurrence_matrix(
            text,
            window_size
        )
    )

    # Generate a co-occurence matrix DataFrame.
    print("Creating co-occurrence matrix DataFrame...")
    cooccurence_matrix_dataframe = (
        pd.DataFrame(
            cooccurrence_matrix,
            index=unique_words,
            columns=unique_words
        )
    )

    # Generate a co-occurrence matrix dictionary where each unique word is a key and the
    # corresponding value is list of co-occurrences with all of the other words.
    print("Converting co-occurrence matrix DataFrame to dictionary...")
    cooccurrence_matrix_dict = cooccurence_matrix_dataframe.to_dict()

    # Convert co-occurrence frequencies into probabilities.
    print("Calculating co-occurrence probabilities...")
    totals, probabilities = (
        functions.helper_functions.cooccurrence_probability.cooccurrence_probability(
            cooccurrence_matrix_dict
        )
    )

    # Convert the probabilities dictionary into a DataFrame.
    print("Converting co-occurrence probabilities dictionary to DataFrame...")
    probabilities = pd.DataFrame.from_dict(
        probabilities,
        orient='index'
    )

    # Convert the probabilities DataFrame into a 2D array.
    print("Converting co-occurrence probabilities DataFrame to 2D array...")
    probabilities = probabilities.to_numpy()


    return unique_words, cooccurrence_matrix_dict, probabilities, text