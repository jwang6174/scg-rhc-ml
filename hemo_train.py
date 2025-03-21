import json
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from hemo_record import NUM_STEPS, RHC_TYPES
from hemo_loader import get_loader

# class AttentionBlock(nn.Module):

#   def __init__(self, in_channels):
#     super(AttentionBlock, self).__init__()
#     self.query = nn.Conv1d(in_channels, in_channels // 8, 1)
#     self.key = nn.Conv1d(in_channels, in_channels // 8, 1)
#     self.value = nn.Conv1d(in_channels, in_channels, 1)
#     self.gamma = nn.Parameter(torch.zeros(1))

#   def forward(self, x):
#     B, C, T = x.size()
#     Q = self.query(x).permute(0, 2, 1) # B x T x C'
#     K = self.key(x) # B x C' x T
#     V = self.value(x) # B x C x T

#     attn = torch.bmm(Q, K) / (K.size(1) ** 0.5) # B x T x T
#     attn = F.softmax(attn, dim=-1)

#     out = torch.bmm(V, attn.permute(0, 2, 1)) # B x C x T
#     out = self.gamma * out + x
#     return out


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


def train(data_name, data_fold):

  model = CardiovascularPredictor()
  optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
  criterion = nn.MSELoss()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  criterion = criterion.to(device)

  filepath = os.path.join('datasets', data_name, f'global_stats_{data_fold}.json')
  with open(filepath, 'r') as f:
    global_segment_stats = json.load(f)

  filepath = os.path.join('datasets', data_name, f'valid_segments_{data_fold}.pkl')
  valid_loader = get_loader(filepath, global_segment_stats, 64)

  filepath = os.path.join('datasets', data_name, f'train_segments_{data_fold}.pkl')
  train_loader = get_loader(filepath, global_segment_stats, 64)

  for epoch in range(100):
    model.train()
    train_loss = 0

    for acc, ecg, bmi, label in train_loader:
      acc = acc.to(device)
      ecg = ecg.to(device)
      bmi = bmi.to(device)
      real_y = label.to(device)

      optim.zero_grad()
      pred_y = model(acc, ecg, bmi)
      loss = criterion(real_y, pred_y)
      loss.backward()
      optim.step()

      train_loss += loss.item()

    model.eval()
    valid_loss = 0
    with torch.no_grad():
      for acc, ecg, bmi, label in valid_loader:
        acc = acc.to(device)
        ecg = ecg.to(device)
        bmi = bmi.to(device)
        real_y = label.to(device)

        pred_y = model(acc, ecg, bmi)
        loss = criterion(real_y, pred_y)
        valid_loss += loss.item()

    train_loss /= len(train_loader)
    valid_loss /= len(valid_loader)
    print(f'Epoch {epoch+1}: '
          f'Train Loss = {train_loss:.4f}, '
          f'Valid Loss = {valid_loss:.4f}')

  print('Training done')


# --- Example usage ---
if __name__ == '__main__':
  data_name = 'hemo_data_1'
  data_fold = '1'
  train(data_name, data_fold)
  
