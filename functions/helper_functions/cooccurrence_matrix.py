import string
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  # Add this import
import matplotlib.colors as mcolors  # Add this import


def create_cooccurrence_matrix(text:str, window_size:int):
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
    text = ''.join([char for char in str(text) if char not in string.punctuation]).lower()

    # Split the text into words.
    words = text.split()
    
    # Remove duplicates from the list of words and sort them alphabetically.
    # This is the vocabulary V.
    unique_words = sorted(set(words))
    print(f'Number of unique words: {len(unique_words)}')

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


def plot_cooccurrence_heatmap(
        unique_words:list,
        cooccurrence_matrix,
        savefig_file_name="cooccurrence_heatmap.png",
        savefig=False,
        show=False
    ):

    """
    Plots a heatmap of the co-occurrence matrix.

    Args:
        unique_words (list): List of unique words.
        cooccurrence_matrix (numpy.ndarray): The co-occurrence matrix.
        savefig_file_name (str): The name of the file to save the heatmap. Default is "cooccurrence_heatmap.png".
    
    The Function:
        1. Converts the co-occurrence matrix to a DataFrame for better visualization.
        2. Caps cooccurrences greater than 5 to 5 on the colorscale.
        3. Creates a heatmap using seaborn.
        4. Saves the heatmap as an image.

    Returns:
        1. None
    """
    # Convert the co-occurrence matrix to a DataFrame for better visualization.
    df = pd.DataFrame(cooccurrence_matrix, index=unique_words, columns=unique_words)

    # Cap cooccurrences greater than 5 to 5.
    # df = df.clip(upper=5)
    df = df.clip(upper=1)

    # Create a mask for zero values.
    mask = (df == 0)

    # Create the heatmap.
    # Set the figure size.
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df,
        cmap="Blues",
        annot=False,
        cbar=True,
        xticklabels=False,
        yticklabels=False,
        mask=mask,
        vmin=0.1,
        vmax=1
    )
    plt.title("Co-occurrence Matrix Heatmap")
    plt.xlabel("Context Words")
    plt.ylabel("Target Words")
    # Rotate x-axis labels for better readability.
    plt.xticks(rotation=90)
    # Adjust layout to prevent clipping.
    plt.tight_layout()

    # Save the heatmap as an image.
    if savefig:
        plt.savefig(savefig_file_name)

    if show:
        plt.show()


def create_cooccurrence_heatmap_from_datafile(
        input_file_name: str,
        output_file_name: str,
        row_range: tuple = None,
        col_range: tuple = None
    ):
    """
    Creates a co-occurrence heatmap from a CSV file, with optional row/column range selection.

    Args:
        1. input_file_name (str): Path to the input CSV or .npy file.
        2. output_file_name (str): File to save the png to.
        3. row_range (tuple, optional): (start, end) indices for rows to plot.
        4. col_range (tuple, optional): (start, end) indices for columns to plot.
    
    The Function:
        1. Reads the CSV file into a DataFrame.
        2. Creates a heatmap using seaborn.
        3. Saves the heatmap as an image.
    
    Returns:
        1. None
    """

    # Read the .npy file into a DataFrame.
    df = pd.DataFrame(np.load(input_file_name, allow_pickle=True).item())

    # Select the specified range of rows and columns if provided.
    if row_range is not None:
        df = df.iloc[row_range[0]:row_range[1], :]
    if col_range is not None:
        df = df.iloc[:, col_range[0]:col_range[1]]

    # Discrete colorbar setup
    vmin = int(df.values.min())
    vmax = int(df.values.max())
    boundaries = np.arange(vmin, vmax + 2) - 0.5  # +2 to include last bin edge
    norm = mcolors.BoundaryNorm(boundaries, ncolors=plt.get_cmap("YlGnBu").N)
    ticks = np.arange(vmin, vmax + 1)

    # Create the heatmap.
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        df,
        cmap="YlGnBu",
        annot=False,
        cbar=True,
        xticklabels=True,
        yticklabels=True,
        norm=norm,
        cbar_kws={
            'ticks': ticks,
            'boundaries': boundaries
        }
    )
    # Center colorbar ticks
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([str(t) for t in ticks])

    plt.title("Co-occurrence Matrix Heatmap")
    plt.xlabel("Context Words")
    plt.ylabel("Target Words")
    # Rotate x-axis labels for better readability.
    plt.xticks(rotation=90)
    # Adjust layout to prevent clipping.
    plt.tight_layout()

    plt.savefig(output_file_name)




######################################################################
# Notes:
# ------
# 1. Need to correct inconsistencies with the desired input data type
#    when making plots.
# 2. Don't need both plot_cooccurrence_heatmap() and
#    create_cooccurrence_heatmap_from_datafile() functions.
######################################################################


# Example usage:
if __name__ == "__main__":
    '''
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
    '''


    # input_file_name = 'testing_scrap_misc/scrap_01/cooccurrence_matrix.npy'
    # output_file_name = 'testing_scrap_misc/scrap_01/cooccurence_matrix.png'
    # row_range = (0, 10)
    # col_range = (0, 10)
    # create_cooccurrence_heatmap_from_datafile(input_file_name, output_file_name, row_range, col_range)