

import subprocess




preprocess = ['python3', 'main.py', '--part', 'preprocess_data']
train_word_vectors = ['python3', 'main.py', '--part', 'train_word_vectors']
train_fnn = ['python3', 'main.py', '--part', 'train_fnn']


subprocess.run(preprocess)

