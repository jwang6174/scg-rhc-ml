import json
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from hemo_loader import get_loader

# Number of samples per batch.
BATCH_SIZE = 64

# Number of epochs.
NUM_EPOCHS = 100


class SelfCrossAttentionBlock(nn.Module):

  def __init__(self, embed_dim=128, num_heads=4):
    super().__init__()
    self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)

  def forward(self, x, cross_input):
    x = x.transpose(1, 2)
    cross_input = cross_input.transpose(1, 2)

    self_attn_out, _ = self.self_attn(x, x, x)
    x = self.norm1(x + self_attn_out)

    cross_attn_out, _ = self.cross_attn(x, cross_input, cross_input)
    x = self.norm2(x + cross_attn_out)

    return x.transpose(1, 2)


class FeatureCNN(nn.Module):

  def __init__(self, in_channels):
    super(FeatureCNN, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv1d(in_channels, 64, kernel_size=5, padding=3),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.MaxPool1d(2),

      nn.Conv1d(64, 128, kernel_size=3, padding=2),
      nn.BatchNorm1d(128),
      nn.ReLU(),
      nn.MaxPool1d(2),

      # nn.Conv1d(128, 256, kernel_size=3, padding=1),
      # nn.BatchNorm1d(256),
      # nn.ReLU(),
      # nn.MaxPool1d(2),
    )

  def forward(self, x):
    x = self.conv(x)
    return x


class CardiovascularPredictor(nn.Module):

  def __init__(self):
    super(CardiovascularPredictor, self).__init__()
    self.acc_net = FeatureCNN(in_channels=3)
    self.ecg_net = FeatureCNN(in_channels=1)
    self.acc_attn = SelfCrossAttentionBlock(embed_dim=128, num_heads=2)
    self.ecg_attn = SelfCrossAttentionBlock(embed_dim=128, num_heads=2)
    self.pool = nn.AdaptiveAvgPool1d(1)

    self.mlp = nn.Sequential(
      nn.Linear(257, 128),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 1),
    )

  def forward(self, acc, ecg, bmi):
    x_acc = self.acc_net(acc)
    x_ecg = self.ecg_net(ecg)
    bmi = bmi.view(-1, 1)

    x_acc = self.acc_attn(x_acc, x_ecg)
    x_ecg = self.ecg_attn(x_ecg, x_acc)

    x_acc = self.pool(x_acc).squeeze(-1)
    x_ecg = self.pool(x_ecg).squeeze(-1)

    x = torch.cat([x_acc, x_ecg, bmi], dim=1)

    out = self.mlp(x)
    return out


def train(model_name, data_name, data_fold):

  # Define learning rate and variable to increment when progressively decreasing
  # the learning rate in case valid loss does not decrease.
  lr = 1e-4
  lr_cnt = 0 
  
  # Define model, optimizer, and criterion.
  model = CardiovascularPredictor()
  optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
  criterion = nn.MSELoss()

  # Move objects to GPU if available.
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  criterion = criterion.to(device)

  # Define best and last checkpoint paths.
  model_best_path = os.path.join('models', model_name, 'model_best.chk')
  model_last_path = os.path.join('models', model_name, 'model_last.chk')

  # Load last checkpoint if exists.
  if os.path.exists(model_last_path):
    checkpoint = torch.load(model_last_path, weights_only=False)
    epoch = checkpoint['epoch'] + 1
    min_valid_loss = checkpoint['min_valid_loss']
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optim_state_dict'])
    lr = checkpoint['lr']
    lr_cnt = checkpoint['lr_cnt']
    for param_group in optim.param_groups:
      param_group['lr'] = lr
  else:
    epoch = 0
    min_valid_loss = float('inf')

  # Load segment stats for value normalization.
  filepath = os.path.join('datasets', data_name, f'global_stats_{data_fold}.json')
  with open(filepath, 'r') as f:
    global_segment_stats = json.load(f)

  # Load valid segments.
  filepath = os.path.join('datasets', data_name, f'valid_segments_{data_fold}.pkl')
  valid_loader = get_loader(filepath, global_segment_stats, BATCH_SIZE)

  # Load train segments.
  filepath = os.path.join('datasets', data_name, f'train_segments_{data_fold}.pkl')
  train_loader = get_loader(filepath, global_segment_stats, BATCH_SIZE)

  # Epoch loop.
  while epoch < NUM_EPOCHS:

    # Training loop.
    model.train()
    train_loss = 0
    for i, (acc, ecg, bmi, label) in enumerate(train_loader, start=1):
      acc = acc.to(device)
      ecg = ecg.to(device)
      bmi = bmi.to(device)
      real_y = label.squeeze(-1).to(device)

      optim.zero_grad()
      pred_y = model(acc, ecg, bmi)
      loss = criterion(real_y, pred_y)
      loss.backward()
      optim.step()

      train_loss += loss.item()

      # Print batch progress message.
      if i % 100 == 0:
        print(f'Epoch {epoch+1}: '
              f'Batch = {i}/{len(train_loader)}, '
              f'Batch Loss = {loss.item()/i:.8f}, '
              f'Epoch Loss = {train_loss/i:.8f} ')

    train_loss /= len(train_loader)

    # Calculate valid loss.
    model.eval()
    valid_loss = 0
    with torch.no_grad():
      for acc, ecg, bmi, label in valid_loader:
        acc = acc.to(device)
        ecg = ecg.to(device)
        bmi = bmi.to(device)
        real_y = label.squeeze(-1).to(device)

        pred_y = model(acc, ecg, bmi)
        loss = criterion(real_y, pred_y)

        valid_loss += loss.item()

    valid_loss /= len(valid_loader)

    # Initialize checkpoint
    checkpoint = {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optim_state_dict': optim.state_dict(),
      'min_valid_loss': min_valid_loss,
      'lr': lr,
      'lr_cnt': lr_cnt,
    }

    print(f'Train Loss = {train_loss:.8f}')
    print(f'Valid Loss = {valid_loss:.8f}')
    print(f'Learning Rate Count = {lr_cnt}')
    print(f'Old Learning Rate = {lr}')

    # If loss improving on valid set, save new best checkpoint.
    if valid_loss < min_valid_loss:
      if valid_loss > train_loss:
        min_valid_loss = valid_loss
        lr_cnt = 0
        checkpoint['lr_cnt'] = lr_cnt
        torch.save(checkpoint, model_best_path)
        print('Saved Best Model')

    # If loss not improving on valid set for certain number of times, then
    # decrease the learning rate.
    else:
      lr_cnt += 1
      if lr_cnt % 5 == 0:
        if lr_cnt >= 25:
          epoch = NUM_EPOCHS
        else:
          lr /= 10
          for param_group in optim.param_groups:
            param_group['lr'] = lr

    print(f'New Learning Rate = {lr}')
    print(f'Best Valid Loss = {min_valid_loss:.8f}')
    print('-' * 20)

    # Save most recent checkpoint.
    torch.save(checkpoint, model_last_path)

    epoch += 1

  print('Training done')


if __name__ == '__main__':
  model_name = sys.argv[1]
  data_name = sys.argv[2]
  data_fold = sys.argv[3]
  train(model_name, data_name, data_fold)
  
