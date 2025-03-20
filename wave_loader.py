import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class SCGDataset(Dataset):

  def __init__(self, segments, segment_size, stats, norm_type):
    """
    Container dataset class.

    Args:
      segments (list[dict]): Segments and info depending on prediction task.
      
      segment_size (int): Number of samples in segment.
      
      stats (dict[str,float]): Various summary statistics for different signals.

      norm_type (str): Normalization type.
    """
    self.segments = self.init_segments(segments, segment_size, stats, norm_type)

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

  def z_score_global(self, signal, signal_avg, signal_std):
    """
    Args:
      signal (ndarray): An (MxN) 2d numpy array, where M is the signal length and
      N is the number of channels.

      signal_avg (float): Signal average.

      signal_std (float); Signal standard deviation.

    Returns:
      signal (ndarray): Z normalized signal with respect to all segments.
    """
    signal = (signal - signal_avg) / signal_std
    return signal

  def minmax_local(self, signal, signal_min, signal_max):
    """
    Args:
      signal (ndarray): An (MxN) 2d numpy array, where M is the signal length and
      N is the number of channels.

      signal_min (float): Local min value.

      signal_max (float): Local max value.

    Returns:
      signal (ndarray): Minmax normalized signal with respective to this segment.
    """
    signal = (signal - signal_min) / (signal_max - signal_min)
    return signal

  def init_segments(self, segments, segment_size, stats, norm_type):
    """
    Args:
      segments (list[dict]): Pre-processed segments.

      segment_size (int): Number of samples in segment.

      stats (dict[str,float]): Various summary statistics for different signals.

      norm_type (str): Normalization type.

    Raises:
      (ValueError) If invalid value passed for norm_type.

    Returns:
      segments (list[dict]): Processed segments.
    """
    for segment in segments:
      if norm_type == 'minmax_local':
        acc = self.minmax_local(segment['acc'], segment['acc_min'], segment['acc_max'])
        rhc = self.minmax_local(segment['rhc'], segment['rhc_min'], segment['rhc_max'])
      elif norm_type == 'z_score_global':
        acc = self.z_score_global(segment['acc'], stats['acc_avg'], stats['acc_std'])
        rhc = self.z_score_global(segment['rhc'], stats['rhc_avg'], stats['rhc_std'])
      else:
        raise ValueError('Invalid value passed for norm_type.')
      acc = self.invert(acc)
      rhc = self.invert(rhc)
      acc = self.pad(acc, segment_size)
      rhc = self.pad(rhc, segment_size)
      segment['acc'] = acc
      segment['rhc'] = rhc
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


def get_loader(segments, segment_size, batch_size, stats, norm_type):
  """
  Args:
    segments (list[ndarray]): List of segments.
    segmentsize (int): Number of samples in segment.
    batch_size (int): Loader batch size.
    stats (dict[str,float]): Various summary statistics for different signals.
    norm_type (str): Normalization type.

  Returns:
    loader (DataLoader): Utility that provides an iterableo over a dataset,
    streamlining the process of feeding data to a model during training or
    evaluation.
  """
  dataset = SCGDataset(segments, segment_size, stats, norm_type)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  return loader
