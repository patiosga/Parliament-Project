import pandas as pd


def load_processed_file(rows = -1):
    file_path = r"processed_data.csv"
    if rows == -1:
        data = pd.read_csv(file_path, encoding='utf-8')
    else:
        data = pd.read_csv(file_path, encoding='utf-8', nrows=rows)

    # Display the first 10 rows of the dataset
    # print("Data Preview:")
    # print(data.head(-1)) 
    return data


def load_raw_file(rows = -1):
    file_path = r"data.csv"
    if rows == -1:
        data = pd.read_csv(file_path, encoding='utf-8')
    else:
        data = pd.read_csv(file_path, encoding='utf-8', nrows=rows)

    # Display the first 10 rows of the dataset
    # print("Data Preview:")
    # print(data.head(-1)) 
    return data