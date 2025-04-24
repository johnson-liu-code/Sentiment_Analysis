
import numpy as np

data_file='project_word_vectors_over_time_03.npy'
word_vectors_over_time = np.load(data_file, allow_pickle=True)

print(word_vectors_over_time[0])