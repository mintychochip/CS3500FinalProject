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
