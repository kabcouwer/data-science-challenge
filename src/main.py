from data.data_loader import load_data
from data.data_processing import preprocess_and_save, preprocess_data
from models.model import Model

def main():
    # Load training data
    raw_data = load_data("./data/raw/Training_DataSet.csv")

    # Process and save data
    output_file_name = "./data/processed/Training_DataSet_Processed.csv"
    processed_data = preprocess_and_save(raw_data, output_file_name)
    # processed_data = preprocess_data(raw_data)
    # Examine processed data
    print("\n Shape of processed data:", processed_data.shape)
    print("\n Processed data: ")
    print(processed_data.head())
    # print(processed_data.describe().T)

    print("\n Training model...")
    # Initialize model
    model = Model(processed_data)
    # Train model
    model.run()

if __name__ == "__main__":
    main()

