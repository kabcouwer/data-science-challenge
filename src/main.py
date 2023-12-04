from data.data_loader import load_data
from data.data_processing import preprocess_data

def main():
    # Load training data
    raw_data = load_data("./data/raw/Training_DataSet.csv")

    # Process data
    data = preprocess_data(raw_data)
    # Show first 5 rows of dataframe
    print(data.info())
    print(data.head())
    print(data.tail())


if __name__ == "__main__":
    main()

