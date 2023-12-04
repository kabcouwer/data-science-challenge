from utils.helper_functions import drop_columns
from utils.helper_functions import transfer_data
from utils.helper_functions import clean_data
from utils.helper_functions import create_car_age_column

def preprocess_data(raw_data):
  # Drop columns using uitility function
  data = drop_columns(raw_data)

  ## Transfer extra data to appropriate columns
  data = transfer_data(data)

  # Remove rows with missing values
  data = data.dropna()

  # Clean up data
  data = clean_data(data)

  # Add car age column
  data = create_car_age_column(data)

  return data
