





def train_nn_on_GloVe_vectors(trained_word_vectors):
    import machine_learning.neural_network_training

    


if __name__ == "__main__":
    import argparse

    import numpy as np
    import pandas as pd

    import machine_learning.glove_vector_training
    # import machine_learning.neural_network_training

    parser_description = "This script contains the code for running the word vector training as well as running the neural network."
    parser = argparse.ArgumentParser(description=parser_description)

    # Parse the command line arguments to tell the script whether to run the GloVe training or to pull data from .npy files that already exist.
    parser.add_argument(
        "--input_file_name",
        default="data/project_data/raw_data/trimmed_training_data.csv",
        help="Path to data used for training.",
        # dest="data_file_name"
    )
    parser.add_argument(
        "--output_file_name",
        default="data/project_data/training_data/test/project_word_vectors_over_time_01.npy",
        help="Path to save word vector training data.",
    )
    parser.add_argument(
        "--train_glove",
        action="store_true",
        help="Run the GloVe training.",
        # dest="run_train_word_vectors"
    )
    parser.add_argument(
        "--save_glove_training_data",
        action="store_true",
        help="Save the trained GloVe vectors to file.",
        # dest="save_data"
    )

    args = parser.parse_args()

    data_file_name = args.input_file_name
    word_vectors_over_time_save_file = args.output_file_name
    run_train_word_vectors = args.train_glove
    save_data = args.save_glove_training_data
    

    # if args.run_train_word_vectors:
    if run_train_word_vectors:
        # print(args.data_file_name)
        word_vectors_over_time = glove_vector_training.GloVe_train_word_vectors(
            data_file_name="data/project_data/raw_data/trimmed_training_data.csv",
            comments_limit=10,
            window_size=64,
            word_vector_length=8,
            save_data=False
        )

    else:
        word_vectors_over_time = np.load(word_vectors_over_time_save_file, allow_pickle=True)

    trained_word_vectors = word_vectors_over_time[-1]
    # print(f'Trained word vector keys:\n{trained_word_vectors.keys()}\n')
    word = 'ayy'
    print(f'Trained word vector for "{word}":\n{trained_word_vectors[word]}\n')

    # data_file_name = "data/project_data/raw_data/trimmed_training_data.csv"
    # comments_limit=10

    # # Get the comments for which we want to train the neural network on.
    # data = pd.read_csv(data_file_name)[:comments_limit]['comment']

    # # print(data)

    # # Split each comment into their component words.
    # words_in_comments = [ comment.split() for comment in data ]

    # # print( words_in_comments )

    # # Throw away any punctuation that are attached to the words.
    # words_in_comments = [
    #     [ word.strip('.,!?()[]{}"\'').lower() for word in comment]
    #     for comment in words_in_comments
    # ]

    # # print( words_in_comments )

    # # Retrieve the vector representation of each word in each comment.
    # vectors_in_comments = [
    #     [ trained_word_vectors[word] for word in comment]
    #     for comment in words_in_comments
    # ]

    # print(len(vectors_in_comments[0][0]))
    # print(vectors_in_comments[1])

    # X = word_vectors_over_time.neural_network_training.frechet_mean(vectors_in_comments)

    # print(X)