import string
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt



def create_cooccurrence_matrix( text, window_size ):
    """
    Creates a co-occurrence matrix from the given text.
    
    Args:
        text (str): The input text from which to create the co-occurrence matrix.
        window_size (int): The size of the context window.
    
    Returns:
        unique_words (list): A list of unique words in the vocabulary.
        cooccurrence_matrix (list of lists representing a 2D matrix): The co-occurrence matrix.
    """
    # Remove punctuation from the text and convert everything to lowercase.
    text = ''.join([char for char in text if char not in string.punctuation]).lower()

    # Split the text into words.
    words = text.split()
    
    # Remove duplicates from the list of words and sort them alphabetically.
    # This is the vocabulary V.
    unique_words = sorted( set(words) )

    # Print the number of unique words in the vocabulary.
    print( f"Number of unique words in the vocabulary: {len(unique_words)}" )

    # Create a dictionary to map each word to its index in the vocabulary.
    word_to_index = { word: index for index, word in enumerate(unique_words) }

    # Initialize the co-occurrence matrix with zeros.
    cooccurrence_matrix = [ [0] * len(unique_words) for _ in range( len(unique_words) ) ]

    # Iterate through each word in the text.
    for i, word in enumerate( words ):

        # Get the index of the current word.
        word_index = word_to_index[ word ]

        # Define the window of context words.
        start_index = max( 0, i - window_size )
        end_index = min( len(words), i + window_size + 1 )

        # Iterate through the context words within the window.
        for j in range( start_index, end_index ):

            if j != i:  # Avoid counting the word itself.
                context_word = words[j]
                context_word_index = word_to_index[context_word]
                # Increment the co-occurrence count for the pair of words.
                cooccurrence_matrix[word_index][context_word_index] += 1

    return unique_words, cooccurrence_matrix


def plot_cooccurrence_heatmap( unique_words, cooccurrence_matrix ):
    """
    Plots a heatmap for the co-occurrence matrix.

    Parameters:
        unique_words (list): List of unique words in the vocabulary.
        cooccurrence_matrix (list of lists): The co-occurrence matrix.
    """
    # Convert the co-occurrence matrix to a DataFrame for better visualization.

    df = pd.DataFrame( cooccurrence_matrix, index = unique_words, columns = unique_words )

    # Create the heatmap.
    plt.figure(figsize=(10, 8))  # Set the figure size.
    sns.heatmap(df, cmap="YlGnBu", annot=False, cbar=True, xticklabels=True, yticklabels=True,
                cbar_kws={'ticks': range(int(df.values.min()), int(df.values.max()) + 1)})
    plt.title("Co-occurrence Matrix Heatmap")
    plt.xlabel("Context Words")
    plt.ylabel("Target Words")
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability.
    plt.tight_layout()  # Adjust layout to prevent clipping.

    plt.savefig("cooccurrence_matrix_heatmap.png")  # Save the heatmap as an image.


def create_cooccurrence_heatmap_from_csv(input_file):
    """
    Reads a co-occurrence matrix from a CSV file and plots a heatmap.

    Parameters:
        input_file (str): Path to the CSV file containing the co-occurrence matrix.
    """

    # Read the CSV file into a DataFrame.
    df = pd.read_csv(input_file, index_col=0)

    # Create the heatmap.
    plt.figure(figsize=(10, 8))  # Set the figure size.
    sns.heatmap(df, cmap="YlGnBu", annot=False, cbar=True, xticklabels=True, yticklabels=True,
                cbar_kws={'ticks': range(int(df.values.min()), int(df.values.max()) + 1)})
    plt.title("Co-occurrence Matrix Heatmap")
    plt.xlabel("Context Words")
    plt.ylabel("Target Words")
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability.
    plt.tight_layout()  # Adjust layout to prevent clipping.

    plt.savefig("cooccurrence_matrix_heatmap.png")  # Save the heatmap as an image.


# Example usage:
if __name__ == "__main__":

    lorem_ipsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc non libero id nunc pharetra tempor non nec magna. Nullam ultrices luctus mauris, eu tincidunt purus vehicula sit amet. Fusce lacinia nibh sed neque pharetra, id lobortis massa porttitor. Nulla id dictum elit, at sollicitudin elit. Vivamus et mollis libero. Morbi vulputate lectus eget pretium fringilla. Nam ac magna non diam rhoncus interdum. Donec posuere posuere pretium. Sed convallis leo sed elit mattis, eget dignissim risus viverra. Phasellus mi erat, egestas non sem elementum, fringilla porttitor sem. Sed mattis laoreet nisi non iaculis. Nam pharetra, velit non egestas rutrum, ante nulla."

    window_size = 6

    unique_words, cooccurrence_matrix = create_cooccurrence_matrix( lorem_ipsum, window_size )

    # Save the co-occurrence matrix to a CSV file.
    output_file = 'cooccurrence_matrix.csv'

    with open( output_file, mode = 'w', newline = '', encoding = 'utf-8' ) as file:
        writer = csv.writer(file)
        
        # Write the header row (unique words).
        writer.writerow( [''] + unique_words )
        
        # Write each row of the matrix with the corresponding word.
        for word, row in zip( unique_words, cooccurrence_matrix ):
            writer.writerow( [word] + row )

    print( f"Co-occurrence matrix saved to {output_file}." )

    input_file = 'cooccurrence_matrix.csv'

    create_cooccurrence_heatmap_from_csv(input_file)