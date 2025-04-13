


import pandas as pd


def cooccurrence_probability(cooccurrence_matrix):
    """
    Calculate the co-occurrence probability of terms in a given co-occurrence matrix.

    Args:
        cooccurrence_matrix (dict): A dictionary representing the co-occurrence matrix.

    Returns:
    """
    
    totals = {}
    probabilities = {}

    for row in cooccurrence_matrix:
        # print(cooccurrence_matrix[row])
        total_count = sum(cooccurrence_matrix[row].values())
        # print(total_count)
        totals[row] = total_count

        row_probabilities = {}

        # Calculate the probability of each term in the row.
        for term, count in cooccurrence_matrix[row].items():
            if total_count > 0:
                row_probabilities[term] = count / total_count
            else:
                row_probabilities[term] = 0


        probabilities[row] = row_probabilities

    return totals, probabilities


# Example usage:
if __name__ == "__main__":
    cooccurrence_csv_file_name = 'data/test_cooccurrence_matrix.csv'

    # Extract the co-occurrence matrix from the CSV file along with the words.
    cooccurrence_matrix = pd.read_csv(cooccurrence_csv_file_name, index_col=0).to_dict()
    terms = list(cooccurrence_matrix.keys())

    totals, probabilities = cooccurrence_probability(cooccurrence_matrix)

    # print(totals)
    # print(probabilities)

    # Save the probabilities to a CSV file.
    probabilities_df = pd.DataFrame(probabilities)
    probabilities_df.to_csv('data/test_cooccurrence_probabilities.csv', index=True, header=True)