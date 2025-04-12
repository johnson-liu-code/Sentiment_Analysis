

import pandas as pd
import keras


if __name__ == "__main__":
    # Define the output CSV file name
    data_file_name = 'data/trimmed_training_data.csv'

    # Read the CSV file into a DataFrame.
    df = pd.read_csv(data_file_name)

    # Ensure all elements in the 'comment' and 'parent_comment' columns are strings.
    # df['comment'] = df['comment'].astype(str)
    # df['parent_comment'] = df['parent_comment'].astype(str)

    


    # model = keras.Sequential()
    # model.add(keras.layers.Dense(64, activation='relu', input_shape=(df.shape[1]-1,)))  # Input layer
    # model.add(keras.layers.Dense(64, activation='relu'))  # Hidden layer
    # model.add(keras.layers.Dense(1, activation='sigmoid'))  # Output layer
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # # Print the model summary to see the architecture.
    # model.summary()
