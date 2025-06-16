




'''
import numpy as np

unique_words_file_name = 'testing_scrap_misc/scrap_01/unique_words.npy'
unique_words = np.load(unique_words_file_name)
print(type(unique_words))
'''


from functions.helper_functions.cooccurrence_matrix import create_cooccurrence_heatmap_from_datafile

create_cooccurrence_heatmap_from_datafile(
    input_file_name = 'testing_scrap_misc/scrap_01/cooccurrence_matrix.npy',
    output_file_name = 'testing_scrap_misc/scrap_01/cooccurence_matrix.png',
    row_range = (150, 199),
    col_range = (250, 299) )
