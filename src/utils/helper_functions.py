import pandas as pd
import numpy as np
from datetime import date

def drop_columns(data):
  # Drop columns
  columns_to_drop = [
    'SellerCity',
    'SellerIsPriv',
    'SellerListSrc',
    'SellerName',
    'SellerRating',
    'SellerRevCnt',
    'SellerState',
    'SellerZip',
    'VehBodystyle',
    'VehCertified',
    'VehColorInt',
    # 'VehDriveTrain',
    'VehEngine',
    'VehFeats',
    'VehFuel',
    'VehHistory',
    'VehListdays',
    'VehPriceLabel',
    'VehSellerNotes',
    'VehType',
    'VehTransmission',
    ]
  return data.drop(columns_to_drop, axis=1)

def transfer_data(data):
  transfer_drive_train_data(data)
  return data

def transfer_data_column(data, column_to_search, search_for, column_to_update, update_with):
  data[column_to_update] = np.where(data[column_to_search].str.contains('|'.join(search_for)), update_with, data[column_to_update])
  return data

def transfer_drive_train_data(data):
  column_to_search = 'Vehicle_Trim'
  column_to_update = 'VehDriveTrain'
   # Convert to lowercase
  data[column_to_search] = data[column_to_search].str.lower()
  # Update values
  transfer_data_column(data, column_to_search, ['4x4', '4wd'], column_to_update, '4WD')
  transfer_data_column(data, column_to_search, ['fwd'], column_to_update, 'FWD')
  transfer_data_column(data, column_to_search, ['awd'], column_to_update, 'AWD')
  transfer_data_column(data, column_to_search, ['rwd'], column_to_update, 'RWD')
  transfer_data_column(data, column_to_search, ['2wd'], column_to_update, '2WD')
  transfer_data_column(data, column_to_search, ['4x2'], column_to_update, '2WD')
  return data

def clean_data(data):
  clean_ext_color_column(data)
  clean_trim_column(data)
  clean_drive_train_column(data)
  return data

def clean_data_column(data, column, search_for, replace_with):
  data[column] = np.where(data[column].str.contains('|'.join(search_for)), replace_with, data[column])
  return data

def clean_ext_color_column(data):
  column = 'VehColorExt'
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
  
  # Delete certain values
  data.drop(data[data[column] == 'Unknown'].index, inplace=True)
  return data

def clean_trim_column(data):
  column = 'Vehicle_Trim'
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
  clean_data_column(data, column, ['luxury'], 'Luxury')
  clean_data_column(data, column, ['platinum'], 'Platinum')

  # Delete certain values
  data.drop(data[data[column] == 'Luxury'].index, inplace=True)
  data.drop(data[data[column] == 'Platinum'].index, inplace=True)
  data.drop(data[data[column] == 'fwd'].index, inplace=True)

  return data

def clean_drive_train_column(data):
  column = 'VehDriveTrain'
  # Convert to lowercase
  data[column] = data[column].str.lower()
  # Replace values
  clean_data_column(data, column, ['awd', 'all'], 'AWD')
  clean_data_column(data, column, ['4wd', '4x4', 'four'], '4WD')
  clean_data_column(data, column, ['fwd', 'front'], 'FWD')
  clean_data_column(data, column, ['2wd', 'two'], '2WD')
  return data

def create_car_age_column(data):
  data['Car_Age'] = date.today().year - data['VehYear']
  return data

  