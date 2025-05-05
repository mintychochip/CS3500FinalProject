import time
from typing import Optional
from typing import List

import pandas as pd

from data_cleaning import clean_data

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

def load_data(file_path: str, train: bool) -> Optional[pd.DataFrame]:
  global choice_bitset
  train_test = 'Training' if train else 'Test'
  print(f'Loading {train_test} data:')
  try:
    start_time = time.time()
    df = pd.read_csv(
        file_path)
    end_time = time.time()
    print(f'{train_test} data loaded in {end_time - start_time:.2f} seconds.')
    if train:
      choice_bitset = [0] * 5
      choice_bitset[0] = 1
    else:
      choice_bitset[3] = 1
    return df
  except FileNotFoundError:
    print(f'Error: {train_test}  data file not found. Please check the path.')
  except pd.errors.ParserError:
    print(
        'Error: Could not parse the CSV file. Please check the file format.')
  except Exception as e:
    print(f'An unexpected error occurred: {e}')


def main():
  previous : Optional[str] = None
  test_df: Optional[pd.DataFrame] = None
  train_df: Optional[pd.DataFrame] = None
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
    for i in range(1, 7):
      choices.append(str(i))

    if choice not in choices:
      print('Invalid option')
      continue

    previous = choice
    if choice == '7':
      print('Exiting Now!')
      break
    if choice == '1':
      train_df = load_data('resources/LA_Crime_Data_2023_to_Present_data.csv',True)
    elif choice_bitset[3] == 1 and choice == '2':
      print('cleaning data')
      if train_df is not None and test_df is not None:
        df = pd.concat([train_df,test_df],axis=0)
        df = clean_data(df,'resources/LA_Crime_Data_2023_to_Present_clean_data.csv')
      else:
        print('Data is not loaded yet.')
      choice_bitset[1] = 1
    elif choice_bitset[1] == 1 and choice == '3':
      print('training nn')
      choice_bitset[2] = 1
    elif choice_bitset[0] == 1 and choice == '4':
      test_df = load_data('resources/LA_Crime_Data_2023_to_Present_test1.csv',False)
    elif choice_bitset[1] == 1 and choice == '5':
      print('generating pred')
      choice_bitset[4] = 1
    elif choice_bitset[4] == 1 and choice == '6':
      print('calculating accuracy')
    else:
      print('Complete the previous step first')


if __name__ == "__main__":
  main()
