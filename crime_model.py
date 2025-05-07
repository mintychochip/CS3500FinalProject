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


def train_nn(df: pd.DataFrame) -> CrimeModel:
  target = 'Target'
  x = df.drop(columns=[target])
  y = df[target]
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
  for epoch in tqdm(range(max_epochs), desc="Training"):
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
      torch.save(model.state_dict(),  CONFIG['weight_path'])
      print(f'Model saved to {CONFIG['weight_path']}')
    else:
      epochs_no_improve += 1
      if epochs_no_improve >= patience:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break

    scheduler.step(val_loss_avg)
    print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
    print_loss_graph(train_losses, val_losses)

  return model

