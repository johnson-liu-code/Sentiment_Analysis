

import pandas as pd



def extract_data( csv_file_name ):
    """
    Extracts the relevant data from the CSV file and returns a DataFrame.

    Args:
        1. csv_file_name (str): The name of the CSV file to extract data from.
    """

    # Load the CSV file into a DataFrame.
    data_all_columns = pd.read_csv( csv_file_name, encoding = "utf-8" )
    # print(f"Data columns: {data_all_columns}")

    # Extract the columns we need.
    data_redux = data_all_columns[ [ "comment", "subreddit", "parent_comment", "label" ] ]

    # Find the number of unique subreddit names.
    # num_unique_subreddits = data_redux["subreddit"].nunique()
    # print(f"Number of unique subreddits: {num_unique_subreddits}")

    # Find the number of rows with missing values.
    num_missing_rows = data_redux.isna().sum()
    # print(f"Number of missing values in each column:\n{ num_missing_rows }")

    # Drop rows with missing values.
    data_redux = data_redux.dropna()
    # num_missing_rows = data_redux.isna().sum()
    # print(f"Number of missing values in each column:\n{ num_missing_rows }")

    return data_redux

# Example usage:
if __name__ == "__main__":
    csv_file_name = "data/sentences.csv"
    data_redux = extract_data( csv_file_name )
    print(data_redux.head())