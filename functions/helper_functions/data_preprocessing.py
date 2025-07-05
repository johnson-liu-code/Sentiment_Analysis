

import sys
import os

# Add the parent directory (i.e., project/) to the path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from word_vector_training_troubleshoot import count_tokens


def filter_comments_by_length(comments, labels, min_len=4, max_len=100):
    """
    Filters comments and corresponding labels based on token count.

    Args:
        comments (list of str): List of text comments.
        labels (list): Corresponding labels.
        min_len (int): Minimum number of words required.
        max_len (int): Maximum number of words allowed.

    Returns:
        filtered_comments (list of str)
        filtered_labels (list)
    """
    filtered_comments = []
    filtered_labels = []

    for comment, label in zip(comments, labels):
        word_count = len(comment.split())
        if min_len < word_count < max_len:
            filtered_comments.append(comment)
            filtered_labels.append(label)

    return filtered_comments, filtered_labels


def data_preprocessing(
        data_file_name: str,
        comments_limit: int,
        window_size: int,
        min_len: int,
        max_len: int
    ):
    """
    Preprocess data for training word vectors using GloVe.

    Args:
        - data_file_name (str): Name of the input file with the training data.
        - comments_limit (int): The number of comments to use to collect words from.
        - window_size (int): Window size for collecting context words.
        - min_len (int): 
        - max_len (int): 

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
    print(f"Number of comments left after dropping NA: {len(data)}...")

    # Collect the text and the corresponding labels from the data.
    # np.ndarray
    text = data['comment'].values
    labels = data['label'].values

    min_len = 4
    max_len = 100

    filtered_comments, filtered_labels = filter_comments_by_length(
        text, labels, min_len, max_len
    )

    print(f"There are {len(filtered_comments)} comments left after filtering out comments that "
            "contain less than {min_len} words and more than {max_len} words...")

    total_tokens, vocab_size, word_freqs = count_tokens.count_total_tokens(filtered_comments)
    print(f"The new dataset has {total_tokens} total tokens.")
    print(f"The new dataset has {vocab_size} unique words.")
    

    # Generate a list of unique words from the text.
    # Generate a co-occurrence matrix from the unique words by scanning through the comments.
    # This returns a 2D array for the co-occurrence matrix.
    print("Generating unique words and co-occurrence matrix...")
    unique_words, cooccurrence_matrix = (
        functions.helper_functions.cooccurrence_matrix.create_cooccurrence_matrix(
            filtered_comments,
            window_size
        )
    )

    # Compute probabilities from the 2D co-occurrence matrix.
    print("Calculating co-occurrence probabilities from the co-occurrnce matrix...")
    row_totals = cooccurrence_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero.
    row_totals[row_totals == 0] = 1
    probabilities = cooccurrence_matrix / row_totals

    # Optionally, create a dictionary if needed elsewhere.
    # cooccurrence_matrix_dict = {
    #     row_word: {col_word: cooccurrence_matrix[i][j] for j, col_word in enumerate(unique_words)}
    #     for i, row_word in enumerate(unique_words)
    # }

    return unique_words, cooccurrence_matrix, probabilities, filtered_comments, filtered_labels

