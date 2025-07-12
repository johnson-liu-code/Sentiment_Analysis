
from collections import defaultdict, Counter
from scipy.sparse import coo_matrix
import numpy as np

def create_sparse_cooccurrence_matrix(
        comments:list,
        window_size:int=10,
        max_vocab_size:int=30000
    ):
    """_summary_
    This expects the sentences in the comments list to already be processed - stopwords
    and punctuation already removed from the sentences.
    ( See functions.data_extraction.get_relevant_data )

    Args:
        comments (list): _description_
        window_size (int, optional): _description_. Defaults to 10.
        max_vocab_size (int, optional): _description_. Defaults to 30000.

    Function:
        ...

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # Build vocabulary
    word_freqs = Counter()
    for sentence in comments:
        words = str(sentence).split()
        word_freqs.update(words)

    most_common = word_freqs.most_common(max_vocab_size)
    vocab = [w for w, _ in most_common]
    word_to_index = {w: i for i, w in enumerate(vocab)}
    vocab_size = len(vocab)

    pair_counts = defaultdict(float)

    for sentence in comments:
        tokens = [w for w in str(sentence).split() if w in word_to_index]
        for i, word in enumerate(tokens):
            wi = word_to_index[word]
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            for j in range(start, end):
                if i == j:
                    continue
                wj = tokens[j]
                wj_index = word_to_index[wj]
                pair_counts[(wi, wj_index)] += 1.0

    if not pair_counts:
        raise ValueError("No co-occurrence pairs found.")

    rows, cols, data = zip(*[(i, j, c) for (i, j), c in pair_counts.items()])
    cooc = coo_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size), dtype=np.float32)
    return vocab, cooc