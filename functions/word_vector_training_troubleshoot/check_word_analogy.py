import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_word_to_idx(vocab, add_special_tokens=False):
    """
    Generate a word-to-index dictionary from a list of unique words.

    Args:
        vocab (List[str]): List of unique words.
        add_special_tokens (bool): Whether to add <PAD> and <UNK> tokens at the start.

    Returns:
        Dict[str, int]: Mapping from word to integer index.
    """
    word_to_idx = {}

    idx = 0

    if add_special_tokens:
        word_to_idx['<PAD>'] = idx
        idx += 1
        word_to_idx['<UNK>'] = idx
        idx += 1

    for word in vocab:
        if word not in word_to_idx:  # skip if already in (e.g. duplicates)
            word_to_idx[word] = idx
            idx += 1
    
    return word_to_idx


def invert_word_index_map(word_to_idx):
    """
    Creates idx_to_word mapping from word_to_idx.

    Args:
        word_to_idx (dict): Mapping from word to index.

    Returns:
        idx_to_word (dict): Mapping from index to word.
    """
    return {idx: word for word, idx in word_to_idx.items()}




def check_word_analogy(word_a, word_b, word_c, embedding_matrix, word_to_idx, idx_to_word=None, top_k=5):
    for word in (word_a, word_b, word_c):
        if word not in word_to_idx:
            raise ValueError(f"Word '{word}' not in vocabulary.")
    
    a = embedding_matrix[word_to_idx[word_a]]
    b = embedding_matrix[word_to_idx[word_b]]
    c = embedding_matrix[word_to_idx[word_c]]
    target_vec = a - b + c

    # Normalize for cosine similarity
    target_vec = F.normalize(target_vec.unsqueeze(0), dim=1)
    all_vecs = F.normalize(embedding_matrix, dim=1)
    
    cosine_sim = torch.matmul(all_vecs, target_vec.T).squeeze()
    top_vals, top_idxs = torch.topk(cosine_sim, top_k)

    results = []
    for idx in top_idxs:
        word = idx_to_word[idx.item()] if idx_to_word else f"#{idx.item()}"
        if word not in {word_a, word_b, word_c}:
            results.append((word, cosine_sim[idx].item()))
        if len(results) == top_k:
            break

    # Plot
    words, scores = zip(*results)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=scores, y=words)
    plt.xlabel("Cosine Similarity")
    plt.title(f"Analogy: {word_a} - {word_b} + {word_c} ≈ ?")
    plt.xlim(0, 1)
    plt.show()

    return results


if __name__ == '__main__':
    trained_word_vectors_file = 'testing_scrap_misc/training_01/word_vector_training/training_logs/weights_epoch_57.pt'
    word_vectors_matrix = torch.load(trained_word_vectors_file)
    
    unique_words_file = 'testing_scrap_misc/training_01/preprocessing/unique_words.npy'
    unique_words = np.load(unique_words_file)

    word_to_idx = generate_word_to_idx(unique_words)
    idx_to_word = invert_word_index_map(word_to_idx)

    # word_list = ['dog', 'cat', 'king', 'queen', 'man', 'woman']
    word_a, word_b, word_c = 'king', 'man', 'queen'

    results = check_word_analogy(   word_a, word_b, word_c, 
                                    word_vectors_matrix,
                                    word_to_idx, idx_to_word,
                                    top_k=10)