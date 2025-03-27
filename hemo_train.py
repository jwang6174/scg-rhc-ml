import json
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from hemo_record import NUM_STEPS, RHC_TYPES
from hemo_loader import get_loader


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

  def __init__(self, in_channels, num_steps):
    super(FeatureCNN, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
      nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.MaxPool1d(2),

      nn.Conv1d(32, 64, kernel_size=5, padding=2),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.MaxPool1d(2),

      nn.Conv1d(64, 128, kernel_size=3, padding=1),
      nn.BatchNorm1d(128),
      nn.ReLU(),
      nn.MaxPool1d(2),
    )

  def forward(self, x):
    x = self.conv(x)
    return x


class CardiovascularPredictor(nn.Module):

  def __init__(self):
    super(CardiovascularPredictor, self).__init__()
    self.acc_net = FeatureCNN(in_channels=3, num_steps=2500)
    self.ecg_net = FeatureCNN(in_channels=1, num_steps=2500)
    self.acc_attn = SelfCrossAttentionBlock(embed_dim=128, num_heads=4)
    self.ecg_attn = SelfCrossAttentionBlock(embed_dim=128, num_heads=4)
    self.pool = nn.AdaptiveAvgPool1d(1)

    self.mlp = nn.Sequential(
      nn.Linear(257, 256),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, 5) # 5 outputs
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


class RMSE_PCC_Loss(nn.Module):

  def __init__(self, alpha=1.0, return_components=False):
    super(RMSE_PCC_Loss, self).__init__()
    self.alpha = alpha
    self.return_components = return_components

  def forward(self, y_pred, y_true):
    y_pred = y_pred.view(y_pred.size(0), -1)
    y_true = y_true.view(y_true.size(0), -1)

    # RMSE
    mse = torch.mean((y_true - y_pred) ** 2, dim=1)
    rmse = torch.sqrt(mse + 1e-8).mean()

    # PCC per output
    vx = y_pred - y_pred.mean(dim=0, keepdim=True)
    vy = y_true - y_true.mean(dim=0, keepdim=True)
    numerator = (vx * vy).sum(dim=0)
    denominator = torch.sqrt((vx ** 2).sum(dim=0)) * torch.sqrt((vy ** 2).sum(dim=0)) + 1e-8
    pcc_per_output = numerator / denominator
    mean_pcc = pcc_per_output.mean()

    loss = rmse + self.alpha * (1 - mean_pcc)

    if self.return_components:
      return loss, rmse.item(), mean_pcc.item()
    return loss


def train(model_name, data_name, data_fold):

  model = CardiovascularPredictor()
  optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
  criterion = RMSE_PCC_Loss(alpha=5.0, return_components=True)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  criterion = criterion.to(device)

  model_best_path = os.path.join('models', model_name, 'model_best.chk')
  model_last_path = os.path.join('models', model_name, 'model_last.chk')

  if os.path.exists(model_last_path):
    checkpoint = torch.load(model_last_path, weights_only=False)
    epoch = checkpoint['epoch'] + 1
    min_valid_loss = checkpoint['min_valid_loss']
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optim_state_dict'])
  else:
    epoch = 0
    min_valid_loss = float('inf')

  filepath = os.path.join('datasets', data_name, f'global_stats_{data_fold}.json')
  with open(filepath, 'r') as f:
    global_segment_stats = json.load(f)

  filepath = os.path.join('datasets', data_name, f'valid_segments_{data_fold}.pkl')
  valid_loader = get_loader(filepath, global_segment_stats, 64)

  filepath = os.path.join('datasets', data_name, f'train_segments_{data_fold}.pkl')
  train_loader = get_loader(filepath, global_segment_stats, 64)

  while epoch < 500:

    model.train()
    train_loss = 0
    train_rmse = 0
    train_pcc = 0
    for acc, ecg, bmi, label in train_loader:
      acc = acc.to(device)
      ecg = ecg.to(device)
      bmi = bmi.to(device)
      real_y = label.to(device)

      optim.zero_grad()
      pred_y = model(acc, ecg, bmi)
      loss, batch_rmse, batch_pcc  = criterion(real_y, pred_y)
      loss.backward()
      optim.step()

      train_loss += loss.item()
      train_rmse += batch_rmse
      train_pcc += batch_pcc

    model.eval()
    valid_loss = 0
    valid_rmse = 0
    valid_pcc = 0
    with torch.no_grad():
      for acc, ecg, bmi, label in valid_loader:
        acc = acc.to(device)
        ecg = ecg.to(device)
        bmi = bmi.to(device)
        real_y = label.to(device)

        pred_y = model(acc, ecg, bmi)
        loss, batch_rmse, batch_pcc = criterion(real_y, pred_y)

        valid_loss += loss.item()
        valid_rmse += batch_rmse
        valid_pcc += batch_pcc

    train_loss /= len(train_loader)
    train_rmse /= len(train_loader)
    train_pcc /= len(train_loader)

    valid_loss /= len(valid_loader)
    valid_rmse /= len(valid_loader)
    valid_pcc /= len(valid_loader)

    print(f'Epoch {epoch+1}: '
          f'Train Loss = {train_loss:.4f}, RMSE = {train_rmse:.4f}, PCC = {train_pcc:.4f} | '
          f'Valid Loss = {valid_loss:.4f}, RMSE = {valid_rmse:.4f}, PCC = {valid_pcc:.4f}')

    if valid_loss < min_valid_loss:
      min_valid_loss = valid_loss
      checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optim.state_dict(),
        'min_valid_loss': min_valid_loss,
      }
      torch.save(checkpoint, model_best_path)
      print('Saved model')
    torch.save(checkpoint, model_last_path)

    epoch += 1

  print('Training done')


if __name__ == '__main__':
  model_name = 'hemo_model_2'
  data_name = 'hemo_data_2'
  data_fold = '1'
  train(model_name, data_name, data_fold)
  
