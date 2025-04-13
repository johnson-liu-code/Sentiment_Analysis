


import pandas as pd


# Define the co-occurrence matrix CSV file name.
matrix_file_name = 'data/test_cooccurrence_matrix.csv'

### THIS CO-OCCURRENCE MATRIX IS FOR TESTING PURPOSES ONLY ###
### IT IS NOT MADE FROM THE TRAINING DATASET ###


# Read the CSV file into a DataFrame.
df = pd.read_csv(matrix_file_name)



data_file_name = 'data/trimmed_training_data.csv'