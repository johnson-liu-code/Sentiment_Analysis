


from extract_data import collect_data




if __name__ == "__main__":

    csv_file_name = "train-balanced-sarcasm.csv"
    data = collect_data( csv_file_name )

    print( data.head() )