


import pandas as pd
import numpy as np
import random


import helper_functions.word_vectors



def g(x, x_max, alpha):
    if x < x_max:
        return (x/x_max) ** alpha
    else:
        return 1.0




# Define the co-occurrence probability matrix CSV file name.
matrix_file_name = 'data/test_cooccurrence_probabilities.csv'
### THIS CO-OCCURRENCE MATRIX IS FOR TESTING PURPOSES ONLY ###
### IT IS NOT MADE FROM THE TRAINING DATASET ###

# Read the CSV file into a DataFrame.
X = pd.read_csv(matrix_file_name)
# print(df.head())

words = X.columns.tolist()
word_vector_size = X.shape[0]
# print(word_vector_size)

word_vectors = helper_functions.word_vectors.create_word_vectors( words, word_vector_size )
new_word_vectors = word_vectors.copy()

# print(word_vectors)
# print(word_vectors[words[0]])
# print(word_vectors[words[1]])

# print(word_vectors[words[2]].T)

# z = np.dot(word_vectors[words[0]], word_vectors[words[1]])
# print(z)

x_max = 100
alpha = 0.75

iter = 100
eta = 0.1

# print(type(X.iloc[0][1]))
X.drop(columns=['Unnamed: 0'], inplace=True)
# print(X)

# print(len(X.columns))

J_over_time = []

for t in range(iter):
    for i in range(len(X.columns)):
        a = np.zeros(word_vector_size)

        for j in range(len(X.columns)):
            if i != j:
                if X.iloc[i][j] != 0:
                    dot_product = np.dot(word_vectors[words[i]], word_vectors[words[j]])

                    # print(i,j)
                    # print(X.iloc[i][j])

                    log_prob = np.log(X.iloc[i][j])
                    g_value = g(X.iloc[i][j], x_max, alpha)
                    
                    a += (dot_product - log_prob) * g_value * word_vectors[words[j]]

        new_word_vectors[words[i]] = word_vectors[words[i]] - eta * 2*a
    
    J = 0
    for i in range(len(X.columns)):
        for j in range(len(X.columns)):
            if i != j:
                if X.iloc[i][j] != 0:
                    dot_product = np.dot(new_word_vectors[words[i]], new_word_vectors[words[j]])
                    log_prob = np.log(X.iloc[i][j])
                    g_value = g(X.iloc[i][j], x_max, alpha)

                    J +=  g_value * (dot_product - log_prob) ** 2
        
    J_over_time.append(J)

    word_vectors = new_word_vectors


# print(new_word_vectors)
# print(word_vectors[words[0]]-new_word_vectors[words[0]])

print(J_over_time)