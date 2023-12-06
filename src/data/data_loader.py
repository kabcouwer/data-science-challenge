import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def load_data_and_convert_to_dataframe(file_path):
    return pd.DataFrame(load_data(file_path))