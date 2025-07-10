
from collections import defaultdict, Counter
from scipy.sparse import coo_matrix

def create_sparse_cooccurrence_matrix(
        comments,
        window_size=10,
        max_vocab_size=30000
    ):
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