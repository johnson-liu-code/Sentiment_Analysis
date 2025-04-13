


import numpy as np


def create_word_vectors(word_list, word_vector_size):
    """
    Create a dictionary of word vectors for the given list of words.
    
    Args:
        word_list (list): A list of words to create vectors for.
        word_vector_size (int): The size of each word vector.
        
    Returns:
        dict: A dictionary mapping each word to its corresponding vector.
    """
    import numpy as np
    
    word_vectors = {}
    
    for word in word_list:
        # Generate a random vector for each word.
        vector = np.random.rand(word_vector_size)
        word_vectors[word] = vector
    
    return word_vectors


# Example usage:
if __name__ == "__main__":
    words = ["apple", "banana", "cherry"]
    vector_size = 5
    vectors = create_word_vectors(words, vector_size)
    
    for word, vector in vectors.items():
        print(f"Word: {word}, Vector: {vector}")