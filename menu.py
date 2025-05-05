import time
from typing import Optional
from typing import List

import pandas as pd

from data_cleaning import clean_data
from crimemodel import train_nn, CrimeModel

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

def load_data(file_path: str) -> Optional[pd.DataFrame]:
  global choice_bitset
  print(f'Loading data:')
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


def main():
  global choice_bitset
  previous : Optional[str] = None
  test_df: Optional[pd.DataFrame] = None
  train_df: Optional[pd.DataFrame] = None
  clean_df: Optional[pd.DataFrame] = None
  model: Optional[CrimeModel] = None
  while True:
    display_menu()
    choice = input("")

    try:
      int(choice)
      print(f'You have entered number: {choice}')
    except ValueError:
      print('Invalid data type for input')
      continue

    if previous is not None and previous == choice:
      print('Do not repeat the same option')
      continue

    choices = []
    for i in range(1, 8):
      choices.append(str(i))

    if choice not in choices:
      print('Invalid option')
      continue

    previous = choice
    if choice == '7':
      print('Exiting Now!')
      break
    if choice == '1':
      print('Loading training data')
      train_df = load_data('resources/LA_Crime_Data_2023_to_Present_data.csv')
      choice_bitset = [0] * 5
      choice_bitset[0] = 1
    elif choice_bitset[3] == 1 and choice == '2':
      print('Cleaning data')
      if train_df is not None and test_df is not None:
        df = pd.concat([train_df,test_df],axis=0)
        start_time = time.time()
        clean_df = clean_data(df,'resources/LA_Crime_Data_2023_to_Present_clean_data.csv')
        end_time = time.time()
        print(
          f'Data cleaned in {end_time - start_time:.2f} seconds.')
      else:
        print('Data is not loaded yet or nil.')
      choice_bitset[1] = 1
    elif choice == '3':
      if clean_df is None:
        try:
          clean_df = load_data(
            'resources/LA_Crime_Data_2023_to_Present_clean_data.csv')
        except Exception as e:
          print(f"Failed to load data: {e}")
          continue
      model = train_nn(clean_df)
      choice_bitset[2] = 1
    elif choice_bitset[0] == 1 and choice == '4':
      print('Loading test data')
      test_df = load_data('resources/LA_Crime_Data_2023_to_Present_test1.csv')
      choice_bitset[3] = 1
    elif choice_bitset[1] == 1 and choice == '5':
      print('generating pred')
      choice_bitset[4] = 1
    elif choice_bitset[4] == 1 and choice == '6':
      print('calculating accuracy')
    else:
      print('Complete the previous step first')


if __name__ == "__main__":
  main()
