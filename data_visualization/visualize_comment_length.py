import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    # Define the output CSV file name.
    data_file_name = '../data/trimmed_training_data.csv'

    # Read the CSV file into a DataFrame.
    df = pd.read_csv(data_file_name)

    # Ensure all elements in the 'comment' and 'parent_comment' columns are strings.
    df['comment'] = df['comment'].astype(str)
    df['parent_comment'] = df['parent_comment'].astype(str)

    # Print out the length of the longest word in the 'comment' column
    # print( df['comment'].map(lambda x: len(x.split())).max() )
    # Print out the row with the longest word in the 'comment' column
    # print( df.loc[df['comment'].map(lambda x: len(x.split())).idxmax()] )

    # Remove rows where the number of words in the 'comment' column is greater than 30.
    df = df[df['comment'].map(lambda x: len(x.split())) <= 30]


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Sarcastic comments.
    text_len = df[df['label'] == 1]['comment'].str.split().map(lambda x: len(x))
    counts, bins, _ = ax1.hist(text_len, color='red', bins=30, edgecolor='black', density=True)
    ax1.set_title('Sarcastic comment')

    # Non-sarcastic comments.
    text_len = df[df['label'] == 0]['comment'].str.split().map(lambda x: len(x))
    counts, bins, _ = ax2.hist(text_len, color='green', bins=30, edgecolor='black', density=True)
    ax2.set_title('Not Sarcastic comment')

    fig.suptitle('Relative Frequency of Words in Comments')
    plt.savefig('words_in_comments.png', dpi=300, bbox_inches='tight')