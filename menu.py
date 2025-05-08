import os
import time
import warnings
from typing import Optional

import pandas as pd
import torch
from torch import Tensor
from torch.nn import Module
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
  accuracy_score, classification_report, roc_auc_score,
  roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from crime_model import train_nn, CrimeModel
from data_cleaning import clean_data
from utils import DEVICE, CONFIG, DF_COLUMNS

# Environment + Warning Config
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
with warnings.catch_warnings():
  warnings.simplefilter('ignore', category=RuntimeWarning)


# ---------- Utility Functions ----------

def display_menu() -> None:
  print('\nMenu:')
  print('    (1) Load training data')
  print('    (2) Process (Clean) data')
  print('    (3) Train NN')
  print('    (4) Load testing data')
  print('    (5) Generate predictions')
  print('    (6) Print results')
  print('    (7) Quit')


def load_data(file_path: str) -> Optional[pd.DataFrame]:
  print(f'Loading data: {file_path}')
  try:
    start = time.time()
    df = pd.read_csv(file_path)
    missing = [col for col in DF_COLUMNS if col not in df.columns]
    if missing:
      print(f'Missing required columns: {missing}')
      return None
    print(f'Data loaded in {time.time() - start:.2f} seconds.')
    return df
  except FileNotFoundError:
    print('Error: File not found.')
  except pd.errors.ParserError:
    print('Error: Could not parse CSV file.')
  except Exception as e:
    print(f'Unexpected error: {e}')
  return None


def try_clean_data(df: Optional[pd.DataFrame], config_path: str, label: str) -> \
Optional[pd.DataFrame]:
  if df is None:
    print(f'No {label} data loaded.')
    return None
  try:
    print(f'Cleaning {label} data...')
    start = time.time()
    cleaned = clean_data(df, config_path)
    print(
      f'{label.capitalize()} data cleaned in {time.time() - start:.2f} seconds.')
    return cleaned
  except Exception as e:
    print(f'Error cleaning {label} data: {e}')
    return None


def generate_predictions(
    model: Optional[Module], clean_train_df: pd.DataFrame,
    clean_test_df: pd.DataFrame
) -> tuple[
  Optional[Tensor], Optional[pd.Series], Optional[list[float]], Optional[
    Module]]:
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
      model.load_state_dict(
        torch.load(CONFIG['weight_path'], map_location=DEVICE))
      model.eval()

    tensor = torch.tensor(x_test_scaled, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
      outputs = model(tensor)
      probs = torch.sigmoid(outputs).cpu().numpy().flatten()
      predictions = torch.tensor((probs >= 0.5).astype(int))
      print('Predictions:', predictions.numpy())
      return predictions, y_test, probs, model

  except Exception as e:
    print(f'Prediction error: {e}')
    return None, None, None, model


def evaluate_model(predictions: Tensor, y_test: pd.Series,
    probs: list[float]) -> None:
  try:
    pred_np = predictions.cpu().numpy()
    acc = accuracy_score(y_test, pred_np)
    print(f'Accuracy: {acc * 100:.2f}%\n')

    print('Classification Report:')
    print(classification_report(y_test, pred_np, digits=4,
                                target_names=['Arrest', 'No Arrest']))

    cm = confusion_matrix(y_test, pred_np)
    print('\nConfusion Matrix:\n', cm)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Arrest', 'No Arrest'],
                yticklabels=['Arrest', 'No Arrest'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    if probs is not None:
      roc_auc = roc_auc_score(y_test, probs)
      fpr, tpr, _ = roc_curve(y_test, probs)
      print(f'ROC AUC Score: {roc_auc:.4f}')
      plt.figure()
      plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
      plt.plot([0, 1], [0, 1], 'k--')
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver Operating Characteristic')
      plt.legend(loc='lower right')
      plt.show()

  except Exception as e:
    print(f'Evaluation error: {e}')


# ---------- Main Program Loop ----------

def main() -> None:
  train_df = clean_train_df = test_df = clean_test_df = None
  model = predictions = y_test = probs_output = None

  while True:
    display_menu()
    choice = input('Enter a menu option: ').strip()

    if choice == '7':
      print('Exiting now!')
      break

    elif choice == '1':
      path = input('Input training data file path: ')
      train_df = load_data(path)

    elif choice == '2':
      clean_train_df = try_clean_data(train_df, CONFIG['clean_train'],
                                      'training')
      clean_test_df = try_clean_data(test_df, CONFIG['clean_test'], 'testing')

    elif choice == '3':
      print('Training model')
      if clean_train_df is None:
        clean_train_df = load_data(CONFIG['clean_train'])
        if clean_train_df is None:
          continue
      model = train_nn(clean_train_df)

    elif choice == '4':
      path = input('Input test data file path: ')
      test_df = load_data(path)

    elif choice == '5':
      print('Generating predictions')
      if clean_train_df is None:
        clean_train_df = load_data(CONFIG['clean_train']) or try_clean_data(
          train_df, CONFIG['clean_train'], 'training')
      if clean_test_df is None:
        clean_test_df = load_data(CONFIG['clean_test']) or try_clean_data(
          test_df, CONFIG['clean_test'], 'testing')
      if clean_train_df and clean_test_df:
        predictions, y_test, probs_output, model = generate_predictions(model,
                                                                        clean_train_df,
                                                                        clean_test_df)

    elif choice == '6':
      print('Calculating Results')
      if predictions is not None and y_test is not None:
        evaluate_model(predictions, y_test, probs_output)
      else:
        print('No predictions available. Run prediction first.')

    else:
      print('Invalid option.')


if __name__ == '__main__':
  try:
    main()
  except Exception as e:
    print(f'Unexpected error occurred: {e}')
