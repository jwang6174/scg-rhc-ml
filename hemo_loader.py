import json
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch
import torch.nn.functional as F
from hemo_record import NUM_STEPS, RHC_TYPE, ACC_CHANNELS, ECG_CHANNELS, IGNORE
from torch.utils.data import Dataset, DataLoader


class HemoDataset(Dataset):

  def __init__(self, segments):
    self.segments = segments

  def add_amplitude_noise(self, signal, noise_level=0.01):
    noise = torch.randn_like(signal) * noise_level
    return signal + noise

  def apply_channel_dropout(self, signal, dropout_prob=0.2):
    if torch.rand(1).item() < dropout_prob:
      channels_to_zero = torch.rand(signal.size(0)) < dropout_prob
      signal[channels_to_zero] = 0.0
    return signal

  def get_noisy_rhc_val(self, rhc_val):
    percent_change = random.randint(1, 10) / 100
    operation = random.choice([1, -1]) * percent_change
    noisy_val = (1 + operation) * rhc_val
    return noisy_val

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
    acc = self.add_amplitude_noise(acc)
    acc = self.apply_channel_dropout(acc)

    ecg = self.pad(self.invert(segment['ecg']))

    weight = self.get_noisy_weight(segment['weight'])
    bmi = weight / ((segment['height'] / 100) ** 2)
    bmi = torch.tensor(bmi, dtype=torch.float32)

    rhc_val = self.get_noisy_rhc_val(segment[RHC_TYPE])
    label = torch.tensor(np.array([rhc_val]), dtype=torch.float32)
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
