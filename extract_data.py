
import pandas as pd


comments_info = pd.read_csv("train-balanced-sarcasm.csv", encoding="utf-8")
# print(comments_info)

print(comments_info.columns)

# Extract the columns we need
comments_info = comments_info[["comment", "label"]]
# find the number of unique subreddits in the dataset