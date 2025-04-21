import string
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  # Add this import


def create_cooccurrence_matrix(text, window_size):
    """
    Creates a co-occurrence matrix for the given text.

    Args:
        1. text (str): The input text.
        2. window_size (int): The size of the context window.

    The Function:
        1. Removes punctuation from the text and converts it to lowercase.
        2. Splits the text into words.
        3. Creates a list of unique words (vocabulary).
        4. Initializes a co-occurrence matrix with zeros.
        5. Iterates through each word in the text and counts co-occurrences within the specified window size.
        6. Returns the unique words and the co-occurrence matrix.
        
    Returns:
        1. unique_words (list): List of unique words in the text.
        2. cooccurrence_matrix (numpy.ndarray): The co-occurrence matrix.
    """
    # Remove punctuation from the text and convert everything to lowercase.
    text = ''.join([char for char in text if char not in string.punctuation]).lower()

    # Split the text into words.
    words = text.split()
    
    # Remove duplicates from the list of words and sort them alphabetically.
    # This is the vocabulary V.
    unique_words = sorted(set(words))

    # Create a dictionary to map each word to its index in the vocabulary.
    word_to_index = {word: index for index, word in enumerate(unique_words)}

    # Initialize the co-occurrence matrix with zeros.
    cooccurrence_matrix = np.zeros((len(unique_words), len(unique_words)), dtype=int)

    # Iterate through each word in the text.
    for i, word in enumerate(words):

        # Get the index of the current word.
        word_index = word_to_index[word]

        # Define the window of context words.
        start_index = max(0, i - window_size)
        end_index = min(len(words), i + window_size + 1)

        # Iterate through the context words within the window.
        for j in range(start_index, end_index):
            # Avoid counting the word itself.
            if j != i:
                context_word = words[j]
                context_word_index = word_to_index[context_word]
                # Increment the co-occurrence count for the pair of words.
                cooccurrence_matrix[word_index][context_word_index] += 1

    return unique_words, cooccurrence_matrix


def plot_cooccurrence_heatmap(unique_words, cooccurrence_matrix, savefig_file_name="cooccurrence_heatmap.png"):
    """
    Plots a heatmap of the co-occurrence matrix.

    Args:
        unique_words (list): List of unique words.
        cooccurrence_matrix (numpy.ndarray): The co-occurrence matrix.
        savefig_file_name (str): The name of the file to save the heatmap. Default is "cooccurrence_heatmap.png".
    
    The Function:
        1. Converts the co-occurrence matrix to a DataFrame for better visualization.
        2. Creates a heatmap using seaborn.
        3. Saves the heatmap as an image.

    Returns:
        1. None
    """
    # Convert the co-occurrence matrix to a DataFrame for better visualization.
    df = pd.DataFrame(cooccurrence_matrix, index=unique_words, columns=unique_words)

    # Create the heatmap.
    # Set the figure size.
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap="YlGnBu", annot=False, cbar=True, xticklabels=True, yticklabels=True,
                cbar_kws={'ticks': np.linspace(df.values.min(), df.values.max(), num=5)})  # Generate 5 evenly spaced ticks
    plt.title("Co-occurrence Matrix Heatmap")
    plt.xlabel("Context Words")
    plt.ylabel("Target Words")
    # Rotate x-axis labels for better readability.
    plt.xticks(rotation=90)
    # Adjust layout to prevent clipping.
    plt.tight_layout()

    # Save the heatmap as an image.
    plt.savefig(savefig_file_name)


def create_cooccurrence_heatmap_from_csv(input_file):
    """
    Creates a co-occurrence heatmap from a CSV file.

    Args:
        1. input_file (str): Path to the input CSV file.
    
    The Function:
        1. Reads the CSV file into a DataFrame.
        2. Creates a heatmap using seaborn.
        3. Saves the heatmap as an image.
    
    Returns:
        1. None
    """
    # Read the CSV file into a DataFrame.
    df = pd.read_csv(input_file, index_col=0)

    # Create the heatmap.
    # Set the figure size.
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap="YlGnBu", annot=False, cbar=True, xticklabels=True, yticklabels=True,
                cbar_kws={'ticks': np.linspace(df.values.min(), df.values.max(), num=5)})  # Generate 5 evenly spaced ticks
    plt.title("Co-occurrence Matrix Heatmap")
    plt.xlabel("Context Words")
    plt.ylabel("Target Words")
    # Rotate x-axis labels for better readability.
    plt.xticks(rotation=90)
    # Adjust layout to prevent clipping.
    plt.tight_layout()

    # Save the heatmap as an image.
    plt.savefig("test02.png")


# Example usage:
if __name__ == "__main__":

    lorem_ipsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc non libero id nunc pharetra tempor non nec magna. Nullam ultrices luctus mauris, eu tincidunt purus vehicula sit amet. Fusce lacinia nibh sed neque pharetra, id lobortis massa porttitor. Nulla id dictum elit, at sollicitudin elit. Vivamus et mollis libero. Morbi vulputate lectus eget pretium fringilla. Nam ac magna non diam rhoncus interdum. Donec posuere posuere pretium. Sed convallis leo sed elit mattis, eget dignissim risus viverra. Phasellus mi erat, egestas non sem elementum, fringilla porttitor sem. Sed mattis laoreet nisi non iaculis. Nam pharetra, velit non egestas rutrum, ante nulla."

    window_size = 6

    unique_words, cooccurrence_matrix = create_cooccurrence_matrix(lorem_ipsum, window_size)

    # Save the co-occurrence matrix to a CSV file.
    output_file = 'cooccurrence_matrix.csv'

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the header row (unique words).
        writer.writerow([''] + unique_words)
        
        # Write each row of the matrix with the corresponding word.
        for word, row in zip(unique_words, cooccurrence_matrix):
            writer.writerow([word] + row)

    print(f"Co-occurrence matrix saved to {output_file}.")

    input_file = 'cooccurrence_matrix.csv'

    create_cooccurrence_heatmap_from_csv(input_file)