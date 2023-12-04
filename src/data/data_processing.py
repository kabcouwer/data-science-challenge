from utils.helper_functions import prepare_data

def preprocess_and_save(raw_data, output_file_name):
  # Prepare data
  processed_data = preprocess_data(raw_data)
  # Save processed data
  processed_data.to_csv(output_file_name, index=False)

  return processed_data

def preprocess_data(raw_data):
  # Process data
  processed_data = prepare_data(raw_data.copy())

  return processed_data
  
