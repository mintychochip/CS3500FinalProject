import os

import numpy as np

# Environment + Warning Config
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
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

with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=RuntimeWarning)

import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from utils import CrimeDataSet, CONFIG, CrimeModel, DEVICE

import torch
import yaml
from typing import List
from torch import nn
from torch.utils.data import Dataset


# Load configuration from YAML
def load_config(file_path='resources/config.yml'):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


# Select GPU if available, else CPU
def get_device() -> torch.device:
    cuda = torch.cuda.is_available()
    selected_device = torch.device('cuda' if cuda else 'cpu')
    if cuda:
        print('GPU Name:', torch.cuda.get_device_name(0))
        print('Total GPU Memory:',
              torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
    else:
        print('Running on CPU.')
    return selected_device


DEVICE: torch.device = get_device()
CONFIG = load_config()

# Relevant dataset columns
DF_COLUMNS: List[str] = [
    'DR_NO', 'Date Rptd', 'DATE OCC', 'TIME OCC', 'AREA', 'AREA NAME',
    'Rpt Dist No', 'Part 1-2', 'Crm Cd', 'Crm Cd Desc', 'Mocodes',
    'Vict Age', 'Vict Sex', 'Vict Descent', 'Premis Cd', 'Premis Desc',
    'Weapon Used Cd', 'Weapon Desc', 'Status', 'Status Desc'
]


class CrimeDataSet(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        self.y = torch.tensor(y.to_numpy(), dtype=torch.float32).to(DEVICE)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class CrimeModel(nn.Module):
    def __init__(self, input_dim):
        super(CrimeModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(CONFIG['dropout_rate']),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(CONFIG['dropout_rate']),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


def scale_features(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[
    pd.DataFrame, pd.DataFrame]:
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled


def create_data_loaders(train_dataset: CrimeDataSet, test_dataset: CrimeDataSet,
                        y_train: pd.Series) -> tuple[DataLoader, DataLoader]:
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    class_sample_counts = [num_neg.item(), num_pos.item()]
    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    sample_weights = weights[y_train.to_numpy().astype(int)]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    batch_size: int = CONFIG['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def compute_roc_auc(model: CrimeModel, test_loader: DataLoader,
                    criterion: _Loss) -> float:
    model.eval()
    val_running_loss = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch.float().view(-1, 1))

            val_running_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu().numpy()
            targets = y_batch.cpu().numpy()
            all_outputs.extend(probs)
            all_targets.extend(targets)
        return roc_auc_score(all_targets, all_outputs)


def print_loss_graph(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.show()


def train_nn(train_df: pd.DataFrame) -> CrimeModel:
    target = 'Target'
    x = train_df.drop(columns=[target])
    y = train_df[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'])

    x_train_scaled, x_test_scaled = scale_features(x_train, x_test)

    train_loader, test_loader = create_data_loaders(
        CrimeDataSet(x_train_scaled, y_train),
        CrimeDataSet(x_test_scaled, y_test),
        y_train)

    criterion = BCEWithLogitsLoss()
    model = CrimeModel(input_dim=x_train_scaled.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CONFIG['learning_rate'],
                                  weight_decay=CONFIG['weight_decay'])

    max_epochs = CONFIG['max_epochs']
    patience = CONFIG['patience']
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=CONFIG['scheduler']['factor'],
        patience=CONFIG['scheduler']['patience'],
        min_lr=CONFIG['scheduler']['min_lr']
    )

    best_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(max_epochs), desc='Training'):
        model.train()
        running_loss = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE).long().view(-1)

            optimizer.zero_grad()
            outputs = model(x_batch).squeeze(1)
            loss = criterion(outputs, y_batch.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE).long().view(-1)
                outputs = model(x_batch).squeeze(1)
                val_loss = criterion(outputs, y_batch.float())
                val_running_loss += val_loss.item()

        val_loss_avg = val_running_loss / len(test_loader)
        val_losses.append(val_loss_avg)

        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            epochs_no_improve = 0
            torch.save(model.state_dict(), CONFIG['weight_path'])
            print(f'Model saved to {CONFIG["weight_path"]}')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break

        scheduler.step(val_loss_avg)
        print(f'Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
        print_loss_graph(train_losses, val_losses)

    return model


import datetime
from typing import List, Optional

import holidays
import pandas as pd
from sklearn.feature_extraction import FeatureHasher


def get_time_of_day_label(dt: datetime) -> Optional[str]:
    # Categorizes the time of day based on the hour
    hour = dt.hour
    time_of_day = {
        (4, 6): 'Dawn',
        (6, 9): 'Morning',
        (9, 12): 'Late Morning',
        (12, 13): 'Noon',
        (13, 17): 'Afternoon',
        (17, 19): 'Evening',
        (19, 21): 'Dusk',
        (21, 24): 'Night',
        (0, 2): 'Midnight',
        (2, 4): 'Twilight'
    }
    for (start, end), label in time_of_day.items():
        if start <= hour < end:
            return label


def get_season_label(dt: datetime) -> Optional[str]:
    # Returns the season corresponding to the date
    season_map = {
        (3, 5): 'Spring',
        (6, 8): 'Summer',
        (9, 11): 'Fall',
        (12, 2): 'Winter'
    }
    for (start, end), season in season_map.items():
        if start <= dt.month <= end or (
                start > end and (dt.month >= start or dt.month <= end)):
            return season


def get_age_group(age: int) -> Optional[str]:
    # Classifies age into defined age groups
    age_groups = {
        (0, 12): 'Child',
        (13, 17): 'Teen',
        (18, 29): 'Young Adult',
        (30, 49): 'Adult',
        (50, 64): 'Middle Aged',
        (65, float('inf')): 'Senior'
    }
    for (start, end), group in age_groups.items():
        if start <= age <= end:
            return group


def convert_status_to_target(df: pd.DataFrame) -> pd.DataFrame:
    # Converts 'Status' column into binary 'Target' for modeling
    df = df[~df['Status'].isin(['CC', 'IC'])].copy()
    df['Target'] = df['Status'].map({'AA': 0, 'JA': 0, 'AO': 1, 'JO': 1})
    df.drop(['Status'], axis=1, inplace=True)
    return df


def apply_one_hot_encoding(df: pd.DataFrame,
                           features: List[str]) -> pd.DataFrame:
    # Applies one-hot encoding to specifiecd categorical features
    return pd.get_dummies(df, columns=features)


def apply_feature_hashing(df: pd.DataFrame, features: List[str],
                          feature_hasher: FeatureHasher) -> pd.DataFrame:
    # Applies feature hashing to high-cardinality categorical features
    for feature in features:
        data_to_hash = df[feature].tolist()
        hashed = feature_hasher.transform(data_to_hash)
        hashed_df = pd.DataFrame(
            hashed.toarray(),
            columns=[f'{feature}_hash_{i}' for i in range(hashed.shape[1])],
            index=df.index
        )
        df = pd.concat([df, hashed_df], axis=1)
    df.drop(columns=features, inplace=True)
    return df


def apply_frequency_encoding(df: pd.DataFrame,
                             features: List[str]) -> pd.DataFrame:
    # Encodes categories based on their relative frequencies
    for col in features:
        freq = df[col].value_counts(normalize=True)
        df[f'{col}_freq'] = df[col].map(freq)
    df.drop(columns=features, inplace=True)
    return df


def filter_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    # Removes outliers using the IQR method
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]


def extract_datetime_components(df: pd.DataFrame) -> pd.DataFrame:
    # Extracts month and day from 'DATE OCC'
    df['DateOccMonth'] = df['DATE OCC'].dt.month
    df['DateOccDay'] = df['DATE OCC'].dt.day
    df.drop(['DATE OCC'], axis=1, inplace=True)
    return df


def reformat_time_column(df: pd.DataFrame) -> pd.DataFrame:
    # Normalizes and converts 'TIME OCC' to datetime format
    df['TIME OCC'] = df['TIME OCC'].fillna('').astype(str)
    if df['TIME OCC'].str.contains(r'\D').any():
        print('Warning: There are non-digit characters in \'TIME OCC\' column.')
    df['TIME OCC'] = df['TIME OCC'].str.zfill(4).str[:-2] + ':' + df[
                                                                      'TIME OCC'].str[
                                                                  -2:]
    df['TIME OCC'] = pd.to_datetime(df['TIME OCC'], format='%H:%M',
                                    errors='coerce')
    return df


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    # Converts specific columns to appropriate data types
    df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
    df['Mocodes'] = df['Mocodes'].astype('string')
    df['Vict Sex'] = df['Vict Sex'].astype('string')
    df['Vict Descent'] = df['Vict Descent'].astype('string')
    df['Status'] = df['Status'].astype('string')
    return df


def rearrange_columns(df: pd.DataFrame, target: str) -> pd.DataFrame:
    # Moves the target column to the end of the DataFrame
    cols = [col for col in df.columns if col != target] + [target]
    return df[cols]


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    # Creates additional features like time of day, season, holidays, etc.
    us_holidays = holidays.US(years=[df['DATE OCC'].dt.year.max()])
    df['TimeOfDay'] = df['TIME OCC'].apply(get_time_of_day_label)
    df['Season'] = df['DATE OCC'].apply(get_season_label)
    df['IsWeekend'] = df['DATE OCC'].apply(lambda dt: dt.weekday() >= 5)
    df['AgeBucket'] = df['Vict Age'].apply(get_age_group)
    df['DayOfWeek'] = df['DATE OCC'].dt.dayofweek
    df['IsHoliday'] = df['DATE OCC'].apply(lambda x: x.date() in us_holidays)
    df['DaysToHoliday'] = df['DATE OCC'].apply(
        lambda x: min(abs((x.date() - h).days) for h in set(us_holidays.keys())))
    df['CrimeCountInArea'] = df.groupby('AREA')['Crm Cd'].transform('count')
    return df


def remove_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Drops columns not useful for modeling and removes duplicates
    cols_to_drop = ['AREA NAME', 'Crm Cd Desc', 'Premis Desc', 'Weapon Desc',
                    'Status Desc', 'Date Rptd', 'Rpt Dist No']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns],
            inplace=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df.drop_duplicates()


def handle_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    codes = ['Crm Cd', 'Premis Cd', 'Weapon Used Cd']
    for code in codes:
        df.loc[df[code].isna(), code] = 0
    df = df[(df['Vict Age'] != 0) & (df['Vict Age'].notna())]
    df = df[(df['Vict Sex'] != 'X') & (df['Vict Sex'] != 'H') & (
        df['Vict Sex'].notna())]
    df = df[(df['Vict Descent'] != '-') & (df['Vict Descent'].notna())]
    df = df.dropna()
    return df


def clean_data(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    # Full pipeline to clean and transform crime data for modeling
    try:
        df = df.set_index('DR_NO')
    except AttributeError:
        raise
    df = remove_unwanted_columns(df)
    df = convert_data_types(df)
    df = handle_invalid_values(df)
    df = filter_outliers(df, 'Vict Age')
    df = reformat_time_column(df)
    df = generate_features(df)
    df.drop(['Vict Age', 'TIME OCC'], axis=1, inplace=True)
    df = extract_datetime_components(df)
    df['Vict Sex'] = df['Vict Sex'].apply(lambda sex: 0 if sex == 'M' else 1)
    df = apply_frequency_encoding(df, ['Crm Cd', 'Premis Cd', 'Weapon Used Cd'])
    df = apply_one_hot_encoding(df, ['AREA', 'Vict Descent',
                                     'TimeOfDay', 'Season', 'AgeBucket',
                                     'DayOfWeek', 'IsHoliday'])
    df['Mocodes'] = df['Mocodes'].fillna('').astype(str).str.split()
    df = apply_feature_hashing(df, ['Mocodes'],
                               FeatureHasher(n_features=64, input_type='string'))
    df = convert_status_to_target(df)
    df = rearrange_columns(df, 'Target')
    try:
        df.to_csv(file_path, index=False)
    except PermissionError:
        raise
    return df


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


def load_data(file_path: str, check_cols: bool = False) -> Optional[
    pd.DataFrame]:
    print(f'Loading data: {file_path}')
    try:
        start = time.time()
        df = pd.read_csv(file_path)
        if check_cols:
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
                   probs: np.ndarray) -> None:
    try:
        pred_np = predictions.cpu().numpy()
        acc = accuracy_score(y_test, pred_np)
        print(f'Accuracy: {acc * 100:.2f}%\n')

        print('Classification Report:')

        print(classification_report(y_test, pred_np,digits=4,target_names=['Arrest','No Arrest']))
        report = classification_report(y_test, pred_np, digits=4,
                                       target_names=['Arrest', 'No Arrest'], output_dict=True)


        accuracy = report['accuracy']
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")

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
        try:
            display_menu()
            choice = input('Enter a menu option: ').strip()

            if choice == '7':
                print('Exiting now!')
                break

            elif choice == '1':
                path = input(f'Input training data file path (enter for default: {CONFIG['train_data']}): ')
                train_df = load_data(path, True) if path != "" else load_data(CONFIG['train_data'], True)

            elif choice == '2':
                load_train: bool = False;
                load_test: bool = False;
                try:
                    if train_df is not None:
                        print('Cleaning train data.')
                        clean_train_df = clean_data(train_df, CONFIG['clean_train'])
                    else:
                        load_train = True
                        print('Load the training data.')
                    if test_df is not None:
                        print('Cleaning test data.')
                        clean_test_df = clean_data(test_df, CONFIG['clean_test'])
                    else:
                        load_test = True
                        print('Load the test data.')
                    if load_test or load_train:
                        continue
                except PermissionError as e:
                    print(f'Permission error: {e}')
                except AttributeError as e:
                    print(f'Attribute error: {e}')
            elif choice == '3':
                if clean_train_df is None:
                    clean_train_df = load_data(CONFIG['clean_train'])
                    if clean_train_df is None:
                        continue
                model = train_nn(clean_train_df)

            elif choice == '4':
                path = input(f'Input test data file path (enter for default: {CONFIG['test_data']}): ')
                test_df = load_data(path, True) if path != "" else load_data(CONFIG['test_data'], True)

            elif choice == '5':
                print('Generating predictions...')
                try:
                    if clean_train_df is None:
                        clean_train_df = load_data(CONFIG['clean_train'])
                        if clean_train_df is None:
                            if train_df is not None:
                                clean_train_df = clean_data(train_df, CONFIG['clean_train'])
                            else:
                                print('Load training data.')
                                continue

                    if clean_test_df is None:
                        clean_test_df = load_data(CONFIG['clean_test'])
                        if clean_test_df is None:
                            if test_df is not None:
                                clean_test_df = clean_data(test_df, CONFIG['clean_test'])
                            else:
                                print('Load testing data.')
                                continue
                except PermissionError as e:
                    print(f'Permission error: {e}')
                except AttributeError as e:
                    print(f'Attribute error: {e}')
                if clean_train_df is not None and clean_test_df is not None:
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
                    print(f'Generated predictions: {predictions}')
            elif choice == '6':
                print('Calculating Results')
                if predictions is not None and y_test is not None:
                    evaluate_model(predictions, y_test, probs)
                else:
                    print('No predictions available. Run prediction first.')

            else:
                print('Invalid option.')
        except Exception as e:
            print(f'Unexpected error occurred: {e}')


if __name__ == '__main__':
    main()
