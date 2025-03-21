import json
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch
import torch.nn.functional as F
from hemo_record import NUM_STEPS, RHC_TYPES, ACC_CHANNELS, ECG_CHANNELS, IGNORE
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 64


class HemoDataset(Dataset):

  def __init__(self, segments):
    self.segments = segments

  def add_time_jitter(self, signal, max_shift=50):
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    if shift == 0:
      return signal
    elif shift > 0:
      return torch.cat([signal[..., shift:], torch.zeros_like(signal[..., :shift])], dim=-1)
    else:
      return torch.cat([torch.zeros_like(signal[..., shift:]), signal[..., :shift]], dim=-1)

  def add_amplitude_noise(self, signal, noise_level=0.01):
    noise = torch.randn_like(signal) * noise_level
    return signal + noise

  def apply_channel_dropout(self, signal, dropout_prob=0.2):
    if torch.rand(1).item() < dropout_prob:
      channels_to_zero = torch.rand(signal.size(0)) < dropout_prob
      signal[channels_to_zero] = 0.0
    return signal

  def get_noisy_hemos(self, segment):
    noisy_vals = {}
    for k, v in segment.items():
      if k in RHC_TYPES:
        if k == 'PCWHR':
          noisy_vals[k] = segment[k]
          continue
        elif k == 'Avg. COmL/min':
          continue
        else:
          percent_change = random.randint(5, 15) / 100
          operation = random.choice([1, -1]) * percent_change
          noisy_vals[k] = (1 + operation) * v
          if k == 'SVmL/beat':
            hr = segment['PCWHR']
            sv = noisy_vals[k]
            noisy_vals['Avg. COmL/min'] = (1 + operation) * sv * hr
    return noisy_vals

  def pad(self, signal):
    if signal.shape[-1] < NUM_STEPS:
      padding = NUM_STEPS - signal.shape[-1]
      signal = F.pad(signal, (0, padding))
    elif signal.shape[-1] > NUM_STEPS:
      signal = signal[:, :, :NUM_STEPS]
    return signal

  def invert(self, signal):
    return torch.tensor(signal.T, dtype=torch.float32)

  def get_noisy_weight(self, real_weight):
    change = random.randint(10, 20) / 10
    noisy_weight = (random.choice([1, -1]) * change) + real_weight
    return noisy_weight

  def __len__(self):
    return len(self.segments)

  def __getitem__(self, index):
    segment = self.segments[index]

    acc = self.pad(self.invert(segment['acc']))
    acc = self.add_time_jitter(acc)
    acc = self.add_amplitude_noise(acc)
    acc = self.apply_channel_dropout(acc)

    ecg = self.pad(self.invert(segment['ecg']))

    weight = self.get_noisy_weight(segment['weight'])
    bmi = weight / ((segment['height'] / 100) ** 2)
    bmi = torch.tensor(bmi, dtype=torch.float32)

    hemos = self.get_noisy_hemos(segment)
    label= torch.tensor([
      hemos['PAM'][0],
      hemos['PCWM'][0],
      hemos['PCWHR'][0],
      hemos['SVmL/beat'][0],
      hemos['Avg. COmL/min'][0]
    ], dtype=torch.float32)

    return acc, ecg, bmi, label


def get_loader(segment_path, global_stats, batch_size):

  # Load segments.
  with open(segment_path, 'rb') as f:
    segments = pickle.load(f)

  for segment in segments:

    # Z-score normalize data.
    for k, v in segment.items():
      if k not in IGNORE:
        segment[k] = (v - global_stats[f'{k}_avg']) / global_stats[f'{k}_std']

  dataset = HemoDataset(segments)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  return loader
