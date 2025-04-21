import pandas as pd
import numpy as np

import helper_functions.cooccurrence_matrix
import helper_functions.cooccurrence_probability
import helper_functions.word_vectors
import machine_learning.gradient_descent



if __name__ == "__main__":

    #################################
    ### This is still in testing. ###
    #################################

    original_data_source = "data/sentences.txt"
    # data_redux = data_extraction.extract_data.extract_data( original_data_source )

    with open(original_data_source, 'r') as file:
        text = file.read()

    window_size = 30

    unique_words, cooccurrence_matrix = (
        helper_functions.cooccurrence_matrix.create_cooccurrence_matrix(
            text, window_size ) )

    # print(unique_words)
    # print(cooccureence_matrix)

    cooccurence_matrix_dataframe = (
        pd.DataFrame(
            cooccurrence_matrix, index = unique_words, columns = unique_words ) )

    # print(cooccurrence_matrix_dataframe)

    cooccurrence_matrix_dict = cooccurence_matrix_dataframe.to_dict()
    # print(cooccurrence_matrix_dict)

    totals, probabilities = (
        helper_functions.cooccurrence_probability.cooccurrence_probability(
            cooccurrence_matrix_dict ) )
    
    # print(totals)
    # print(probabilities[unique_words[0]])

    # savefig_file_name = "cooccurrence_probability_heatmap.png"

    # helper_functions.cooccurrence_matrix.plot_cooccurrence_heatmap(
    #     unique_words, probabilities, savefig_file_name )
    
    word_vectors = (
        helper_functions.word_vectors.create_word_vectors( 
            unique_words, len(unique_words) ) )

    new_word_vectors = word_vectors.copy()

    x_max = 100
    alpha = 0.75

    iter = 100
    eta = 0.1

    # print(type(X.iloc[0][1]))

    # Convert the dictionary to a DataFrame
    probabilities = pd.DataFrame.from_dict(probabilities, orient='index')
    # print(probabilities)

    # probabilities.drop(columns=['Unnamed: 0'], inplace=True)
    # print(X)

    # print(len(X.columns))

    J_over_time, word_vectors_over_time = machine_learning.gradient_descent.descent(
        unique_words, word_vectors, new_word_vectors, probabilities, x_max, alpha, eta, iter )

    # word_vectors_over_time = []
    # word_vectors_over_time.append(word_vectors)

    # J_over_time = []

    # for t in range(iter):
    #     for i in range(len(probabilities.columns)):
    #         a = np.zeros(len(unique_words))

    #         for j in range(len(probabilities.columns)):
    #             if i != j:
    #                 if probabilities.iloc[i][j] != 0:
    #                     dot_product = np.dot(word_vectors[unique_words[i]], word_vectors[unique_words[j]])

    #                     # print(i,j)
    #                     # print(X.iloc[i][j])

    #                     log_prob = np.log(probabilities.iloc[i][j])
    #                     g_value = g(probabilities.iloc[i][j], x_max, alpha)
                        
    #                     a += (dot_product - log_prob) * g_value * word_vectors[unique_words[j]]

    #         new_word_vectors[unique_words[i]] = word_vectors[unique_words[i]] - eta * 2*a
        
    #     J = 0
    #     for i in range(len(probabilities.columns)):
    #         for j in range(len(probabilities.columns)):
    #             if i != j:
    #                 if probabilities.iloc[i][j] != 0:
    #                     dot_product = np.dot(new_word_vectors[unique_words[i]], new_word_vectors[unique_words[j]])
    #                     log_prob = np.log(probabilities.iloc[i][j])
    #                     g_value = g(probabilities.iloc[i][j], x_max, alpha)

    #                     J +=  g_value * (dot_product - log_prob) ** 2
            
    #     J_over_time.append(J)

    #     word_vectors = new_word_vectors.copy()
    #     word_vectors_over_time.append(word_vectors)

    # print(new_word_vectors)
    # print(word_vectors[words[0]]-new_word_vectors[words[0]])

    # print(J_over_time)

    # Save J_over_time to binary file
    J_over_time_save_file = 'J_over_time_02.npy'
    np.save(J_over_time_save_file, J_over_time)

    # Save word_vectors_over_time to binary file
    word_vectors_over_time_save_file = 'word_vectors_over_time_02.npy'
    np.save(word_vectors_over_time_save_file, word_vectors_over_time)