

import extract_data


import csv
import pandas as pd
import nltk
# Make sure to download the stopwords corpus if you haven't already.
# nltk.download('stopwords') or use python3 -m nltk.downloader stopwords in the terminal.
import string



if __name__ == "__main__":
    # Extract data
    data = extract_data.extract_data( csv_file_name = 'train-balanced-sarcasm.csv' )

    # Define the output CSV file name
    output_file = 'training_data.csv'

    # Write data to CSV
    data.to_csv( output_file, index = False )

    # Open .csv file and extract data into dataframe using pandas.
    csv_file_name = 'training_data.csv'
    data = pd.read_csv( csv_file_name, encoding = "utf-8" )
    
    # Define words and punctuations to remove from the text in the data.
    stopwords_list = set( nltk.corpus.stopwords.words( 'english' ) )
    punctuation = list( string.punctuation )
    stopwords_list.update( punctuation )

    # Remove punctionations from individual words in the 'comment' and 'parent_comment' columns.
    data['comment'] = data['comment'].apply( lambda x: ''.join( [ char for char in x if char not in punctuation ] ) )
    data['parent_comment'] = data['parent_comment'].apply( lambda x: ''.join( [ char for char in x if char not in punctuation ] ) )
    
    # Remove stop words from the text in 'comment' and 'parent_comment' columns.
    data['comment'] = data['comment'].apply( lambda x: ' '.join( [ word.lower() for word in x.split() if word.lower() not in stopwords_list ] ) )
    data['parent_comment'] = data['parent_comment'].apply( lambda x: ' '.join( [ word.lower() for word in x.split() if word.lower() not in stopwords_list ] ) )

    # Ensure all elements in the 'comment' and 'parent_comment' columns are strings.
    # data['comment'] = data['comment'].astype(str)
    # data['parent_comment'] = data['parent_comment'].astype(str)

    # Define the output CSV file name.
    output_file = 'trimmed_training_data.csv'

    # Write data to CSV.
    data.to_csv( output_file, index = False )