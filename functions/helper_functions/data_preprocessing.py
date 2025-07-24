

def filter_comments_by_length(
        comments: list,
        labels: list,
        min_len: int = 4,
        max_len: int = 100
    ):

    filtered_comments = []
    filtered_labels = []
    for comment, label in zip(comments, labels):
        word_count = len(comment.split())
        if min_len <= word_count <= max_len:
            filtered_comments.append(comment)
            filtered_labels.append(label)
    return filtered_comments, filtered_labels



def data_preprocessing(
        data_file_name: str,
        comments_limit: int,
        window_size: int,
        min_len: int,
        max_len: int,
        max_vocab_size: int,
        data_file_name: str,
        comments_limit: int,
        window_size: int,
        min_len: int,
        max_len: int,
        max_vocab_size: int
    ):
    """
    Preprocess data for training word vectors using GloVe.

    Args:
        - data_file_name (str): Name of the input file with the training data.
        - comments_limit (int): The number of comments to use to collect words from.
        - window_size (int): Window size for collecting context words.
        - min_len (int): 
        - max_len (int): 
        - max_vocab_size (int): 
        - min_len (int): 
        - max_len (int): 
        - max_vocab_size (int): 

    Returns:
        - unique_words (list): List of unique words.
        - cooccurrence_matrix_dict (dict): Dictionary of co-occurrence matrices.
        - probabilities (np.ndarray): 2D array of co-occurrence probabilities.
    """
    
    ###########################################################################
    import pandas as pd
    import numpy as np
    ###########################################################################

    # Read the data into a DataFrame.
    # Randomly pick the data points.
    print(f"Reading data from {data_file_name} with a limit of {comments_limit} comments...")
    data = pd.read_csv(data_file_name).sample(n=comments_limit, random_state=94)
    # data = pd.read_csv(data_file_name)[:comments_limit]

    data = data.dropna(subset=['comment'])
    print(f"Number of comments left after dropping NA: {len(data)}...")
    print(f"Number of comments left after dropping NA: {len(data)}...")

    # Collect the text and the corresponding labels from the data.
    # np.ndarray
    text = data['comment'].values
    labels = data['label'].values

    filtered_comments, filtered_labels = filter_comments_by_length(
        text, labels, min_len, max_len
    )

    print(f"There are {len(filtered_comments)} comments left after filtering out comments that "
          f"contain less than {min_len} words and more than {max_len} words...")


    # There's got to be better way of doing this ...
    import sys
    import os

    # Add the parent directory (i.e., project/) to the path.
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

    import count_tokens
    import cooccurrence_matrix

    total_tokens, vocab_size, word_freqs = count_tokens.count_total_tokens(filtered_comments)
    print(f"The new dataset has {total_tokens} total tokens.")
    print(f"The new dataset has {vocab_size} unique words.")
    
    print(f"Generating unique words and co-occurrence matrix using the most frequent {max_vocab_size} words...")

    unique_words, cooc_matrix_sparse = cooccurrence_matrix.create_sparse_cooccurrence_matrix(
            filtered_comments,
            window_size,
            max_vocab_size
        )
    filtered_comments, filtered_labels = filter_comments_by_length(
        text, labels, min_len, max_len
    )

    print(f"There are {len(filtered_comments)} comments left after filtering out comments that "
          f"contain less than {min_len} words and more than {max_len} words...")


    # There's got to be better way of doing this ...
    import sys
    import os

    # Add the parent directory (i.e., project/) to the path.
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

    import count_tokens
    import cooccurrence_matrix

    total_tokens, vocab_size, word_freqs = count_tokens.count_total_tokens(filtered_comments)
    print(f"The new dataset has {total_tokens} total tokens.")
    print(f"The new dataset has {vocab_size} unique words.")
    
    print(f"Generating unique words and co-occurrence matrix using the most frequent {max_vocab_size} words...")

    unique_words, cooc_matrix_sparse = cooccurrence_matrix.create_sparse_cooccurrence_matrix(
            filtered_comments,
            window_size,
            max_vocab_size
        )

    return unique_words, cooc_matrix_sparse, filtered_comments, filtered_labels