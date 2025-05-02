





def train_nn_on_GloVe_vectors(trained_word_vectors):
    import machine_learning.neural_network_training

    


if __name__ == "__main__":
    import argparse

    import numpy as np
    import pandas as pd

    import helper_functions.frechet_mean

    import machine_learning.glove_vector_training
    # import machine_learning.neural_network_training

    parser_description = "This script contains the code for running the word vector training as well as running the neural network."
    parser = argparse.ArgumentParser(description=parser_description)

    # Parse the command line arguments to tell the script whether to run the GloVe training or to pull data from .npy files that already exist.
    parser.add_argument(
        "--input_file_name",
        default="data/project_data/raw_data/trimmed_training_data.csv",
        help="Path to data used for training.",
    )
    parser.add_argument(
        "--output_file_name",
        default="data/project_data/training_data/test/project_word_vectors_over_time_01.npy",
        help="Path to save word vector training data.",
    )
    parser.add_argument(
        "--load_file_name",
        default="test_trained_word_vectors.npy",
        help="Path to load saved word vector training data.",
    )
    parser.add_argument(
        "--train_glove",
        action="store_true",
        default=False,
        help="Run the GloVe training.",
    )
    parser.add_argument(
        "--save_glove_training_data",
        action="store_true",
        default=False,
        help="Save the trained GloVe vectors to file.",
    )

    args = parser.parse_args()

    data_file_name = args.input_file_name
    word_vectors_over_time_save_file = args.output_file_name
    load_file_name = args.load_file_name
    run_train_word_vectors = args.train_glove
    save_data = args.save_glove_training_data

    if run_train_word_vectors:
        word_vectors_over_time = machine_learning.glove_vector_training.GloVe_train_word_vectors(
            data_file_name="data/project_data/raw_data/trimmed_training_data.csv",
            comments_limit=10,
            window_size=64,
            word_vector_length=8,
            save_data=False,
            x_max = 100,
            alpha = 0.75,
            iter = 100,
            eta = 0.1
        )

        np.save(
            word_vectors_over_time_save_file,
            word_vectors_over_time
        )

    else:
        word_vectors_over_time = np.load(
            load_file_name,
            allow_pickle=True
        )

    trained_word_vectors = word_vectors_over_time[-1]
 
    # print(f'Trained word vectors: {trained_word_vectors}')

    comments_limit = 64
    data = pd.read_csv(data_file_name)[:comments_limit]
    comments = data['comment'].values
    labels = data['label'].values

    comments = [ comment.split() for comment in comments ]

    # Remove words in each comment that does not appear in the trained_word_vectors dictionary
    comments = [
        [ word for word in comment if word in trained_word_vectors
        ] for comment in comments
    ]

    # print(f'Comments: {comments}')


    # vectorized_comments = [
    #     [ trained_word_vectors.get(
    #             word,
    #             np.zeros_like( next( iter( trained_word_vectors.values() ) ) )
    #         ) for word in comment
    #     ] for comment in comments
    # ]

    vectorized_comments = [
        [ trained_word_vectors[word] for word in comment
        ] for comment in comments
    ]

    # print(f'Vectorized comments: {vectorized_comments}')

    # print(f'Word vector keys:\n{trained_word_vectors.keys()}\n')
    # print(f'Trained word vectors: {trained_word_vectors}')
 
    # i = 1
    # print(f'Comment {i}: {comments[i]}')
    # print(f'Vectorized comment {i}: {vectorized_comments[i]}')

    # print(len(vectorized_comments))

    centered_comments = helper_functions.frechet_mean.frechet_mean(
        vectorized_comments,
        word_vector_length = 8
    )
    # print(f"Frech'ed comments: {centered_comments}")
    # print(len(centered_comments))

    