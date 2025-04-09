
import pandas as pd



def collect_data( csv_file_name ):
    # Load the CSV file into a DataFrame.
    # data_all_columns = pd.read_csv("train-balanced-sarcasm.csv", encoding="utf-8")
    data_all_columns = pd.read_csv( csv_file_name, encoding = "utf-8" )
    # print(data_all_columns)
    # print(f"Data columns: {data_all_columns}")

    # Extract the columns we need.
    data_redux = data_all_columns[ [ "comment", "subreddit", "parent_comment", "label" ] ]

    # Dind the number of unique subreddit names.
    # num_unique_subreddits = data_redux["subreddit"].nunique()
    # print(f"Number of unique subreddits: {num_unique_subreddits}")

    return data_redux