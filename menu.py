import os
import warnings

from sklearn.preprocessing import MinMaxScaler

with warnings.catch_warnings():
  warnings.simplefilter("ignore", category=RuntimeWarning)

from sklearn.metrics import accuracy_score, classification_report

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
from typing import Optional

import pandas as pd
import torch
from crime_model import train_nn, CrimeModel
from data_cleaning import clean_data
from utils import DEVICE, CONFIG

def display_menu():
  print("Menu:")
  print("    (1) Load training data")
  print("    (2) Process (Clean) data")
  print("    (3) Train NN")
  print("    (4) Load testing data")
  print("    (5) Generate Predictions")
  print("    (6) Print Accuracy (Actual Vs Predicted)")
  print("    (7) Quit")

def load_data(file_path: str) -> Optional[pd.DataFrame]:
  print(f'Loading data: {file_path}')
  try:
    start_time = time.time()
    df = pd.read_csv(file_path)
    end_time = time.time()
    print(f'Data loaded in {end_time - start_time:.2f} seconds.')
    return df
  except FileNotFoundError:
    print(f'Error: File not found. Please check the path.')
  except pd.errors.ParserError:
    print('Error: Could not parse the CSV file. Please check the file format.')
  except Exception as e:
    print(f'An unexpected error occurred: {e}')
  return None

def main():
  train_df = None
  clean_train_df = None
  test_df = None
  clean_test_df = None
  model = None
  predictions = None
  y_test = None

  while True:
    display_menu()
    choice = input("Enter a menu option:").strip()

    if not choice.isdigit() or choice not in map(str, range(1, 8)):
      print('Invalid option')
      continue

    if choice == '7':
      print('Exiting Now!')
      break

    if choice == '1':
      path = input("Input training data file path:")
      train_df = load_data(path)

    elif choice == '2':
      print('Cleaning data:')
      try:
        if train_df is not None:
          clean_train_df = clean_data(train_df, 'resources/clean_train.csv')
          print('Training data cleaned.')
        if test_df is not None:
          clean_test_df = clean_data(test_df, 'resources/clean_test.csv')
          print('Testing data cleaned.')
      except PermissionError as e:
        print(f"Permission Error: {e}")
      except AttributeError as e:
        print(f"Attribute error: {e}")
      except Exception as e:
        print(f"Data cleaning error: {e}")

    elif choice == '3':
      if clean_train_df is None:
        clean_train_df = load_data(CONFIG['clean_train'])
        if clean_train_df is None:
          continue
      model = train_nn(clean_train_df)

    elif choice == '4':
      path = input("Input test data file path:")
      test_df = load_data(path)

    elif choice == '5':
      print('Generating Predictions')
      if clean_train_df is None:
        clean_train_df = load_data(CONFIG['clean_train'])
        if clean_train_df is None:
          continue

      if clean_test_df is None:
        clean_test_df = load_data(CONFIG['clean_test'])
        if clean_test_df is None:
          continue

      try:
        target = 'Target'
        x_train = clean_train_df.drop(columns=[target])
        x_test = clean_test_df.drop(columns=[target])
        y_test = clean_test_df[target]

        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_test_scaled = scaler.transform(x_test)

        if model is None:
          model = CrimeModel(input_dim=x_test_scaled.shape[1]).to(DEVICE)
          model.load_state_dict(torch.load(CONFIG['weight_path'], map_location=DEVICE))
          model.eval()

        tensor = torch.tensor(x_test_scaled, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
          outputs = model(tensor)
          probs = torch.sigmoid(outputs).cpu().numpy().flatten()
          predictions = torch.tensor((probs >= 0.5).astype(int))
          print('Predictions:', predictions.numpy())

      except Exception as e:
        print(f'Prediction error: {e}')

    elif choice == '6':
      print('Calculating accuracy')
      if predictions is not None and y_test is not None:
        try:
          predictions_cpu = predictions.cpu().numpy()
          accuracy = accuracy_score(y_test, predictions_cpu)
          print(f'Accuracy: {accuracy * 100:.2f}%')
          print('\nClassification Report:')
          print(classification_report(y_test, predictions_cpu, digits=4))
        except Exception as e:
          print(f'Error in calculating accuracy: {e}')
      else:
        print('No predictions available. Run prediction first.')

if __name__ == "__main__":
  main()
