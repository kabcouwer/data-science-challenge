import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

class Model:
    def __init__(self, data):
      self.data = data
      self.model = None

    def preprocess_data(self):
      # Apply log transformation to 'VehMileage'
      self.data['VehMileage_log'] = np.log1p(self.data['VehMileage'])
      # Apply log transformation to 'Dealer_Listing_Price'
      self.data['Dealer_Listing_Price_log'] = np.log1p(self.data['Dealer_Listing_Price'])

    def split_data(self):
      # Include 'VehMileage_log' and 'Dealer_Listing_Price_log' in X
      X = self.data.drop(['Dealer_Listing_Price', 'VehMileage', 'Dealer_Listing_Price_log'], axis=1)
      y = self.data['Dealer_Listing_Price_log']
      return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
      self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
      self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
      predictions = self.model.predict(X_test)
      r2 = r2_score(y_test, predictions)
      print(f"R2 Score with Log-Transformed Feature: {r2}")

    def make_predictions(self):
      # Make predictions on the entire dataset
      predictions = np.expm1(self.model.predict(self.data.drop(['Dealer_Listing_Price', 'VehMileage', 'Dealer_Listing_Price_log'], axis=1)))

      # Create a DataFrame with index, original 'Dealer_Listing_Price', and predictions
      result_df = pd.DataFrame({
          'Index': self.data.index,
          'Original_Dealer_Listing_Price': self.data['Dealer_Listing_Price'],
          'Predicted_Dealer_Listing_Price': predictions
      })

      return result_df

    def export_predictions(self, filepath='data/processed/Test_DataSet_Processed.csv'):
      # Ensure the directory exists
      os.makedirs(os.path.dirname(filepath), exist_ok=True)
      
      # Export predictions DataFrame to a CSV file
      predictions_df = self.make_predictions()
      predictions_df.to_csv(filepath, index=False)
      print(f"Predictions exported to {filepath}")

    def run(self):
      self.preprocess_data()  # Apply log transformation
      X_train, X_test, y_train, y_test = self.split_data()
      self.train_model(X_train, y_train)
      self.evaluate_model(X_test, y_test)

      # Export predictions to CSV after running the model
      print("\n Exporting predictions...")
      self.export_predictions()