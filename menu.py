import os
import warnings

with warnings.catch_warnings():
  warnings.simplefilter("ignore", category=RuntimeWarning)

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
from typing import Optional

import pandas as pd
import numpy as np
import torch
from crime_model import train_nn, CrimeModel, scale_features
from data_cleaning import clean_data
from utils import DEVICE, CONFIG

choice_bitset = [0] * 5


def display_menu():
  print("Menu:")
  print("    (1) Load training data")
  print("    (2) Process (Clean) data")
  print("    (3) Train NN")
  print("    (4) Load testing data")
  print("    (5) Generate Predictions")
  print("    (6) Print Accuracy (Actual Vs Predicted)")
  print("    (7) Quit")


def columns_have_equal_length(df: pd.DataFrame) -> bool:
  lengths = [len(df[col]) for col in df.columns]
  first_length = lengths[0]
  for col, length in zip(df.columns, lengths):
    if length != first_length:
      return False
  return True


def load_data(file_path: str) -> Optional[pd.DataFrame]:
  print(f'Loading data: {file_path}')
  try:
    start_time = time.time()
    df = pd.read_csv(
        file_path)
    end_time = time.time()
    print(f'Data loaded in {end_time - start_time:.2f} seconds.')
    return df
  except FileNotFoundError:
    print(f'Error: File not found. Please check the path.')
    raise
  except pd.errors.ParserError:
    print(
        'Error: Could not parse the CSV file. Please check the file format.')
    raise
  except Exception as e:
    print(f'An unexpected error occurred: {e}')
    raise

def safe_load_data(path: str) -> Optional[pd.DataFrame]:
  try:
    return load_data(path)
  except (FileNotFoundError, pd.errors.ParserError) as e:
    print(f"Error loading data: {e}")
  except Exception as e:
    print(f"Unexpected error while loading data: {e}")
  return None


def safe_clean_data(df: pd.DataFrame, path: str) -> Optional[pd.DataFrame]:
  try:
    return clean_data(df, path)
  except PermissionError as e:
    print(f"Permission error: {e}")
  except Exception as e:
    print(f"Failed to clean data: {e}")
  return None


def main():
  train_df: Optional[pd.DataFrame] = None
  test_df: Optional[pd.DataFrame] = None
  clean_df: Optional[pd.DataFrame] = None
  model: Optional[CrimeModel] = None
  predictions: Optional[torch.Tensor] = None

  y_test = None
  while True:
    display_menu()
    choice = input("Enter a menu option:").strip()

    if not choice.isdigit():
      print('Invalid data type for input')
      continue
    if choice not in map(str, range(1, 8)):
      print('Invalid option')
      continue

    if choice == '7':
      print('Exiting Now!')
      break

    if choice == '1':
      path = input("Input training data file path:")
      train_df = safe_load_data(path)

    elif choice == '2':
      print('Cleaning data:')
      if train_df is not None and test_df is not None:
        if not (
            columns_have_equal_length(train_df) and columns_have_equal_length(
            test_df)):
          print("Mismatch in column lengths. Aborting.")
          continue

        df = pd.concat([train_df, test_df], axis=0)
        start_time = time.time()
        clean_df = safe_clean_data(df, 'resources/clean.csv')
        end_time = time.time()
        if clean_df is not None:
          print(f'Data cleaned in {end_time - start_time:.2f} seconds.')
      else:
        print('Data is not loaded yet or nil.')

    elif choice == '3':
      if clean_df is None:
        clean_df = safe_load_data(CONFIG['clean_file_path'])
        if clean_df is None:
          continue
      model = train_nn(clean_df)

    elif choice == '4':
      path = input("Input test data file path:")
      test_df = safe_load_data(path)

    elif choice == '5':
      print('Generating Predictions')
      if clean_df is None:
        clean_df = safe_load_data(CONFIG['clean_file_path'])
        if clean_df is None:
          continue

      target = 'Target'
      x = clean_df.drop(columns=[target])
      y = clean_df[target]
      x_train, x_test, y_train, y_test = train_test_split(
          x, y, test_size=CONFIG['test_size'],
          random_state=CONFIG['random_state']
      )
      x_train_scaled, x_test_scaled = scale_features(x_train, x_test)

      if model is None:
        try:
          model = CrimeModel(input_dim=x_train_scaled.shape[1]).to(DEVICE)
          model.load_state_dict(
            torch.load(CONFIG['weight_path'], map_location=DEVICE))
          model.eval()
        except Exception as e:
          print(f"Failed to load model: {e}")
          continue

      try:
        tensor = torch.tensor(x_test_scaled, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
          outputs = model(tensor)
          probs = torch.sigmoid(outputs).cpu().numpy().flatten()
          predictions = torch.tensor((probs >= 0.5).astype(int))
          print('Predictions:', predictions.numpy())
      except Exception as e:
        print(f'Prediction error: {e}')
        continue

    elif choice == '6':
      print('Calculating accuracy')
      if predictions is not None:
        try:
          predictions_cpu = predictions.cpu().numpy()
          accuracy = accuracy_score(y_test, predictions_cpu)
          print(f'Accuracy: {accuracy * 100:.2f}%')
          print('\nClassification Report:')
          print(classification_report(y_test, predictions_cpu, digits=4))
        except Exception as e:
          print(f'Error in calculating accuracy: {e}')
      else:
        print('Complete the previous step first')


if __name__ == "__main__":
  try:
    main()
  except Exception as e:
    print(f'Some unexpected error occurs: {e}')
