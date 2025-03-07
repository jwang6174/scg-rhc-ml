import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class SCGDataset(Dataset):

  def __init__(self, segments, segment_size):
    """
    Container dataset class.

    Args:
      segments (list[dict]): Segments and info depending on prediction task.
      segment_size (int): Number of samples in segment.
    """
    self.segments = self.init_segments(segments, segment_size)

  def pad(self, signal, segment_size):
    """
    Args:
      signal (ndarray): An (MxN) 2d numpy array, where M is the signal length and
      N is the number of channels.

      segment_size (float): Segment duration (sec).

    Returns:
      signal (ndarray): Pad or trim signal to appropriate size.
    """
    if signal.shape[-1] < segment_size:
      padding = segment_size - signal.shape[-1]
      signal = F.pad(signal, (0, padding))
    elif signal.shape[-1] > segment_size:
      signal = signal[:, :, :segment_size]
    return signal

  def invert(self, signal):
    """
    Args:
      signal (ndarray): An (MxN) 2d numpy array, where M is the signal length and
      N is the number of channels.

    Returns:
      tensor (Tensor): Signal with M and N swapped.
    """
    signal = torch.tensor(signal.T, dtype=torch.float32)
    return signal

  def minmax_norm(self, signal):
    """
    Args:
      signal (ndarray): An (MxN) 2d numpy array, where M is the signal length and
      N is the number of channels.

    Returns:
      signal (ndarray): Min-max normalized signal.
    """
    min = np.min(signal)
    max = np.max(signal)
    signal = (signal - min) / (max - min + 0.0001)
    return signal, min, max

  def init_segments(self, segments, segment_size):
    """
    Args:
      segments (list[dict]): Pre-processed segments.
      segment_size (int): Number of samples in segment.

    Returns:
      segments (list[dict]): Processed segments.
    """
    for segment in segments:
      acc, acc_min, acc_max = self.minmax_norm(segment['acc'])
      rhc, rhc_min, rhc_max = self.minmax_norm(segment['rhc'])
      acc = self.invert(acc)
      rhc = self.invert(rhc)
      acc = self.pad(acc, segment_size)
      rhc = self.pad(rhc, segment_size)
      segment['acc'] = acc
      segment['rhc'] = rhc
      segment['acc_min'] = acc_min
      segment['acc_max'] = acc_max
      segment['rhc_min'] = rhc_min
      segment['rhc_max'] = rhc_max
    return segments

  def __len__(self):
    """
    Returns:
      (int): Number of segments.
    """
    return len(self.segments)

  def __getitem__(self, index):
    """
    Args:
      index (int): Segment index.

    Returns:
      (dict): Segment at given index.
    """
    return self.segments[index]


def get_loader(segments, segment_size, batch_size):
  """
  Args:
    segments (list[ndarray]): List of segments.
    segmentsize (int): Number of samples in segment.
    batch_size (int): Loader batch size.

  Returns:
    loader (DataLoader): Utility that provides an iterableo over a dataset,
    streamlining the process of feeding data to a model during training or
    evaluation.
  """
  dataset = SCGDataset(segments, segment_size)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  return loader
