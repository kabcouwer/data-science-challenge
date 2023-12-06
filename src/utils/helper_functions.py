import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

def prepare_data(data):
  # Reassign index
  reassign_index(data)
  # Add car age column
  data = create_car_age_column(data)
  # Add location column
  # data = create_location_column(data)
  # Drop columns using uitility function
  data = drop_columns(data)
  # Remove rows with missing values
  data = data.dropna()
  # Clean up data
  data = clean_data(data)
  # Handle categorical data
  data = handle_categorical_data(data)
  # Bin numeric columns
  data = bin_numeric_columns(data)
  # Remove outliers of numeric columns
  data = remove_outliers(data)

  return data

# Cleaning functions

def reassign_index(data):
  data.set_index('ListingID', inplace=True)

  return data

def create_car_age_column(data):
  data['Car_Age'] = date.today().year - data['VehYear']
  # Drop VehYear column
  data.drop('VehYear', axis=1, inplace=True)
  
  return data

def create_location_column(data):
  data['SellerLocation'] = data['SellerCity'] + ', ' + data['SellerState']
  # Drop SellerCity and SellerState columns
  data.drop(['SellerCity', 'SellerState'], axis=1, inplace=True)
  
  return data

def drop_columns(data):
  # Drop columns
  columns_to_drop = [
    'SellerListSrc',
    'SellerName',
    'SellerRevCnt',
    'SellerState',
    'SellerCity',
    'SellerZip',
    'VehBodystyle',
    'VehCertified',
    'VehColorInt',
    'VehEngine',
    'VehFeats',
    'VehListdays',
    'VehModel',
    'VehSellerNotes', # TO DO
    'VehType',
    'VehTransmission',
    'Vehicle_Trim'
    ]

  return data.drop(columns_to_drop, axis=1)

def clean_data(data):
  clean_ext_color_column(data)
  # clean_trim_column(data)
  clean_drive_train_column(data)
  clean_history_column(data)
  clean_engine_column(data)
  clean_fuel_column(data)
  
  return data

def clean_data_column(data, column, search_for, replace_with):
  data[column] = np.where(data[column].str.contains('|'.join(search_for)), replace_with, data[column])
  
  return data

def clean_ext_color_column(data):
  column = 'VehColorExt'
  # Confirm column exists
  if column in data.columns:
    # Convert to lowercase
    data[column] = data[column].str.lower()
    # Replace values
    clean_data_column(data, column, ['yellow'], 'Yellow')
    clean_data_column(data, column, ['beige', 'beigh'], 'Beige')
    clean_data_column(data, column, ['orange'], 'Orange')
    clean_data_column(data, column, ['green'], 'Green')
    clean_data_column(data, column, ['red', 'sangria', 'velvet', 'deep amethyst', 'maroon', 'burgundy'], 'Red')
    clean_data_column(data, column, ['white', 'pearl', 'ivory'], 'White')
    clean_data_column(data, column, ['blue'], 'Blue')
    clean_data_column(data, column, ['gray', 'grey', 'gy', 'shadow metallic', 'bronze dune', 'steel', 'charcoal', 'granite', 'dark gray', 'rhino clearcoat', 'granite crystal metallic', 'granite chrystal metallic', 'granite crystal clearcoat metallic', 'dark granite metallic', 'maximum steel'], 'Gray')
    clean_data_column(data, column, ['purple'], 'Purple')
    clean_data_column(data, column, ['silver', 'sil', 'diamond', 'billet'], 'Silver')
    clean_data_column(data, column, ['black', 'midnight sky'], 'Black')
    clean_data_column(data, column, ['brown', 'mocha'], 'Brown')
    clean_data_column(data, column, ['gold', 'tan', 'pewter', 'platinum', 'cashmere'], 'Gold')
    clean_data_column(data, column, ['bronze'], 'Bronze')
    clean_data_column(data, column, ['pink'], 'Pink')
    clean_data_column(data, column, ['other', 'not specified', 'unspecified', 'unknown', 'certified'], 'Unknown')
    
    # Convert unknow to nan
    data[column] = np.where(data[column] == 'Unknown', np.nan, data[column])
  
  return data

def clean_trim_column(data):
  column = 'Vehicle_Trim'
  # Confirm column exists
  if column in data.columns:
    # Convert to lowercase
    data[column] = data[column].str.lower()
    # Replace values
    clean_data_column(data, column, ['75th anniversary'], '75th Anniversary')
    clean_data_column(data, column, ['srt'], 'SRT')
    clean_data_column(data, column, ['limited x'], 'Limited X')
    clean_data_column(data, column, ['limited'], 'Limited')
    clean_data_column(data, column, ['laredo e'], 'Laredo E')
    clean_data_column(data, column, ['upland'], 'Upland')
    clean_data_column(data, column, ['overland'], 'Overland')
    clean_data_column(data, column, ['high altitude'], 'High Altitude')
    clean_data_column(data, column, ['altitude'], 'Altitude')
    clean_data_column(data, column, ['trackhawk'], 'Trackhawk')
    clean_data_column(data, column, ['trailhawk'], 'Trailhawk')
    clean_data_column(data, column, ['summit'], 'Summit')
    clean_data_column(data, column, ['laredo'], 'Laredo')
    clean_data_column(data, column, ['sterling edition'], 'Sterling Edition')
    clean_data_column(data, column, ['base'], 'Base')
    clean_data_column(data, column, ['premium'], 'Premium Luxury')
    clean_data_column(data, column, ['luxury'], 'Luxury')
    clean_data_column(data, column, ['platinum'], 'Platinum')
    clean_data_column(data, column, ['sport'], 'Sport')

    # Delete certain values
    data.drop(data[data[column] == 'fwd'].index, inplace=True)

  return data

def clean_drive_train_column(data):
  column = 'VehDriveTrain'
  # Confirm column exists
  if column in data.columns:
    # Convert to lowercase
    data[column] = data[column].str.lower()
    # Replace values
    clean_data_column(data, column, ['awd', 'all'], 'AWD')
    clean_data_column(data, column, ['4wd', '4x4', 'four'], '4WD')
    clean_data_column(data, column, ['fwd', 'front'], 'FWD')
    clean_data_column(data, column, ['2wd', 'two'], '2WD')
  
  return data

def clean_history_column(data):
  column = 'VehHistory'
  # Confirm column exists
  if column in data.columns:
    # Convert to lowercase
    data[column] = data[column].str.lower()
    # Create new columns
    create_number_of_owners_column(data)
    create_accident_column(data)
    # Drop VehHistory column
    data.drop('VehHistory', axis=1, inplace=True)

  return data

def clean_engine_column(data):
  column = 'VehEngine'
  # Confirm column exists
  if column in data.columns:
    # Convert to lowercase
    data[column] = data[column].str.lower()
    # Replace values
    clean_data_column(data, column, ['v6', 'v-6', 'v 6', '6 cyl', '3.6', '6', '3.0 l'], 'V6')
    clean_data_column(data, column, ['v8', 'v-8', 'v 8', '8 cylinder', '8 cyl', '8-cyl', '5.7', 'hemi'], 'V8')

  return data

def clean_fuel_column(data):
  column = 'VehFuel'
  # Confirm column exists
  if column in data.columns:
    # Convert Unknown to nan
    data[column] = np.where(data[column] == 'Unknown', np.nan, data[column])

  return data

def create_number_of_owners_column(data):
  length = len(data['VehHistory'])
  # data['Num_Owners'] = data['VehHistory'].str.extract(r'(\d*\.\d+|\d+)', expand=False)
  data['Num_Owners'] = data.VehHistory.apply(lambda x: history_number_of_owners_conditional(x))
  
  return data

def create_accident_column(data):
  length = len(data['VehHistory'])

  data['Accident'] = data.VehHistory.apply(lambda x: history_accident_conditional(x))
  
  return data

def history_accident_conditional(history_string):
  if 'accident' in history_string:
    return 1
  else:
    return 0

def history_number_of_owners_conditional(history_string):
  if 'owner' in history_string:
    return history_string.split()[0]
  else:
    return np.nan

def handle_categorical_data(data):
  # Convert categorical data to numerical data
  data = map_column(data, 'VehMake')
  data = dummy_column(data, ['VehColorExt', 'VehFuel', 'VehDriveTrain', 'VehPriceLabel'])

  return data

def map_column(data, column):
  dict = {}
  values = data[column].unique()

  for index, value in enumerate(values):
    dict[value] = index
  
  data[column] = data[column].map(dict)
  
  return data

def dummy_column(data, columns):
  data = pd.get_dummies(data, columns=columns, drop_first=True)
  
  return data

def bin_numeric_columns(data):
  # SellerRating bins
  column = 'SellerRating'
  bins = [0, 3, 4, 4.5, 5]
  labels = ['very poor', 'poor', 'average', 'good']
  data = bin_column(data, column, bins, labels)
  data = dummy_column(data, [column])
  
  return data

def bin_column(data, column, bins, labels):
  data[column] = pd.cut(data[column], bins=bins, labels=labels)
  
  return data

# Statistical functions

def remove_outliers(data, threshold=3):
  # Find z-scores of data
  z_scores_dataframe = find_z_scores(data, threshold)
  # Remove outliers
  data = data[(z_scores_dataframe < threshold).all(axis=1)]

  return data

def find_z_scores(data, threshold=3):
  # Columns to check for outliers
  numeric_columns = data.select_dtypes(include=[np.number])
  # Find z-scores
  z_scores = np.abs(stats.zscore(numeric_columns))
  # Create dataframe of z-scores
  z_scores_dataframe = pd.DataFrame(z_scores, index=numeric_columns.index, columns=numeric_columns.columns)

  return z_scores_dataframe

# Plotting functions

def plot_data(data, feature1, feature2):
  plt.scatter(data[feature1], data[feature2])
  plt.title(feature1 + " vs " + feature2 + " Scatter Plot")
  plt.xlabel(feature1)
  plt.ylabel(feature2)
  plt.show()

def plot_data_with_bins(data, feature1, feature1_bins, feature2):
  colors = np.digitize(data[feature1], bins=feature1_bins)

  plt.scatter(data[feature1], data[feature2], c=colors, cmap='rainbow')
  plt.colorbar(label='Bins')
  plt.title(feature1 + " vs " + feature2 + " Scatter Plot")
  plt.xlabel(feature1)
  plt.ylabel(feature2)
  plt.show()

def plot_boxplot(data, feature1):
  sns.set(style="whitegrid")
  ax = sns.boxplot(x=data[feature1])

  return ax
  