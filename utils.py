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
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(CONFIG['dropout_rate']),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
