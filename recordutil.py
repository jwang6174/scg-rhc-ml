import json
import numpy as np
import os
import pandas as pd
import pickle
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
from datetime import datetime
from itertools import combinations
from pathlib import Path
from pathutil import DATABASE_PATH
from sklearn.model_selection import train_test_split
from time import time
from timeutil import timelog
from torch.utils.data import Dataset, DataLoader
from noiseutil import has_noise

# Signal sample rate.
SAMPLE_RATE = 500


class SCGDataset(Dataset):

  def __init__(self, segments, segment_size):
    """
    Container dataset class.

    Args:
      segments (list[dict]): Segments with patch signals, patient attributes, and 
      RHC value.

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

  def __len__(self):
    return len(self.segments)

  def __getitem__(self, index):
    return self.segments[index]

  def minmax_norm(self, signal):
    """
    Args:
      signal (ndarray): An (MxN) 2d numpy array, where M is the signal length and
      N is the number of channels.

    Returns:
      signal (ndarray): Signal min-max normalized.
    """
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 0.0001)
    return signal

  def invert(self, signal):
    """
    Args:
      signal (ndarray): An (MxN) 2d numpy array, where M is the signal length and
      N is the number of channels.

    Returns:
      tensor (Tensor): Inverted signal as tensor object.
    """
    tensor = torch.tensor(signal.T, dtype=torch.float32)
    return tensor

  def init_segments(self, segments, segment_size):
    """
    Args:
      segments (list[dict]): List of segments prior to minmax normalization.
      Note that minmax normalizations is relative to each segment.

      segment_size (int): Number of samples in segment.

    Returns:
      segments (list[dict]): Segments after inter-segment minmax
      normalization.
    """
    for segment in segments:
      segment['acc'] = self.pad(self.invert(self.minmax_norm(segment['acc'])), segment_size)
      if 'ecg' in segment:
        segment['ecg'] = self.pad(self.invert(self.minmax_norm(segment['ecg'])), segment_size)
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


def get_record_names(path):
  """
  Args:
    path (str): SCG-RHC database path.

  Returns:
    names (list[str]): Record names.
  """
  names = set()
  for filename in os.listdir(os.path.join(path, 'processed_data')):
    if filename.endswith('.dat') or filename.endswith('.hea'):
      names.add(Path(filename).stem)
  names = sorted(list(names))
  return names


def get_record_names_with_challenge(path):
  """
  Args:
    path (str): SCG-RHC database path.

  Returns:
    with_challenge (list[str]): Record names with physiologic challenge 
    (eg, nitro, dobutamine, fluids, leg raise, etc).
  """
  with_challenge = []
  for name in get_record_names(path):
    with open(os.path.join(path, 'processed_data', f'{name}.json'), 'r') as f:
      data = json.load(f)
      if data['IsChallenge'] == 1:
        with_challenge.append(name)
  return with_challenge


def get_record_names_without_challenge(path):
  """
  Args:
    path (str): SCG-RHC database path.

  Returns:
    no_challenge (list[str]): Record names without physiologic challenge.
  """
  all_names = get_record_names(path)
  with_challenge = get_record_names_with_challenge(path)
  no_challenge = sorted(list(set(all_names) - set(with_challenge)))
  return no_challenge


def get_patient_code(record_name):
  """
  Args:
    record_name (str): Record name that includes patient code.

  Returns:
    code (str): Patient code.
  """
  code = record_name.split('-')[0]
  return code


def get_record_names_of_multiple_caths(path):
  """
  Args:
    path (str): SCG-RHC database path.

  Returns:
    multi (list[str]): Record names of multiple caths.
  """
  multi = []
  counts = {}
  record_names = get_record_names(path)
  
  # Get record count for each patient.
  for name in record_names:
    code = get_patient_code(name)
    counts[code] = counts.get(code, 0) + 1

  # Get record names with multiple caths.
  for name in record_names:
    code = get_patient_code(name)
    if counts[code] > 1:
      multi.append(name)

  return multi


def get_record_names_of_single_caths(path):
  """
  Args:
    path (str): SCG-RHC database path.

  Returns:
    single (list[str]): Record names of only one cath.
  """
  single = []
  all_names = get_record_names(path)
  multi_names = get_record_names_of_multiple_caths(path)
  single = sorted(list(set(all_names) - set(multi_names)))
  return single


def get_channels(db_path, record_name, channel_names, start, stop):
  """
  Args:
    db_path (str): SCG-RHC databse path.
    record_name (str): Record name.
    channel_names (list[str]): Channel names.
    start (int): Start index for record segment.
    stop (int): Stop index for record segment.

  Raises:
    IndexError: If channel name cannot be found in record.

  Returns:
    channels (ndarray): An (MxN) 2d numpy array, where M is the signal length and
    N is the number of channels.
  """
  
  # Load record.
  record = wfdb.rdrecord(os.path.join(db_path, 'processed_data', record_name))
  
  # Get channel indexes.
  channel_indexes = []
  for name in channel_names:
    if name in record.sig_name:
      channel_indexes.append(record.sig_name.index(name))
    else:
      raise IndexError(f'Channel {name} not found in record {record_name}')

  # Get channels for specific segment.
  channels = record.p_signal[start:stop, channel_indexes]

  return channels


def get_value_test_codes_and_record_names(path):
  """
  Get all patient codes and record names that may be included in a test set. To 
  maximize available training data, test set should only include patients that 
  underwent 1 cath and did not undergo physiologic challenge. This schema is
  only used for value prediction.

  Args:
    path (str): SCG-RHC database path.

  Returns:
    patient_codes (list[str]): Patient codes in test set.
    record_names (list[str]): Record names in test set.
  """
  set1 = set(get_record_names_without_challenge(path))
  set2 = set(get_record_names_of_single_caths(path))
  record_names = list(set1.intersection(set2))
  patient_codes = list(set([get_patient_code(x) for x in record_names]))
  return patient_codes, record_names


def get_waveform_test_codes_and_record_names(path):
  """
  Get all patient codes and record names that may be included in a test set for
  waveform prediction. Unlike for value prediction, all records are candidates
  to be in the test set.

  Args:
    path (str): SCG-RHC database path.

  Returns:
    patient_codes (list[str]): Patient codes in test set.
    record_names (list[str]): Record names in test set.
  """
  record_names = get_record_names(path)
  patient_codes = list(set([get_patient_code(x) for x in record_names]))
  return patient_codes, record_names 


def get_train_codes_and_record_names(path, test_record_names):
  """
  Get patient codes and record names to be included in a train set by subtracting
  test record names from all record names.

  Args:
    path (str): SCG-RHC database path.
    test_record_names (list[str]):  Record names in test set.

  Returns:
    patient_codes (list[str]): Patient codes in train set.
    record_names (list[str]): Record names in train set.
  """
  record_names = sorted(list(set(get_record_names(path)) - set(test_record_names)))
  patient_codes = sorted(list(set([get_patient_code(x) for x in record_names])))
  return patient_codes, record_names


def get_sample_index(start_time, sample_time, sample_rate):
  """
  Args:
    start_time (datetime): Start time without date.
    sample_time (datetime): Sample time without date.
    sample_rate (int): Signal sample rate.

  Returns:
    sample_index (int): Sample index corresponding to given sample time.
  """
  diff = (sample_time - start_time).total_seconds()
  sample_index = int(diff * sample_rate)
  return sample_index


def get_start_and_stop_times(path, record_name):
  """
  Args:
    path (str): SCG-RHC database path.
    record_name (str): Record name.

  Returns:
    start (datetime): Recording start time without date.
    stop (datetime): Recording stop time without date.
  """
  with open(os.path.join(path, 'processed_data', f'{record_name}.json'), 'r') as f:
    data = json.load(f)
    start = datetime.strptime(data['MacStTime'], '%d-%b-%Y %H:%M:%S')
    stop = datetime.strptime(data['MacEndTime'], '%d-%b-%Y %H:%M:%S')
    return start, stop


def get_last_chamber_time(path, record_name):
  """
  Args:
    path (str): SCG-RHC database path.
    record_name (str): Record name.

  Returns:
    time_obj (datetime) Last chamber time without date.
  """
  with open(os.path.join(path, 'processed_data', f'{record_name}.json'), 'r') as f:
    data = json.load(f)
    last_time = None
    for chamber, time_str in data['ChamEvents'].items():
      time_obj = datetime.strptime(time_str.strip(), '%d-%b-%Y %H:%M:%S %p')
      if last_time is None or time_obj > last_time:
        last_time = time_obj
    return last_time


def get_logged_challenge_time(path, record_name):
  """
  Get start of recording after physiologic challenge presumably in full effect,
  as logged in PAM_PCWP_timestamp_in_TBME.json, assuming that PA_VI represents
  the start.

  Args:
    path (str): SCG-RHC database path.
    record_name (str): Record name.

  Raises:
    IndexError: If record name not in PAM_PCWP_timestamp_in_TBME.json, suggesting
    that no challenge was recorded in the given record.

  Returns:
    time_obj (datetime): Time without date of when effects of physiologic
    challenge were presumably in full effect.
  """
  with open(os.path.join(path, 'meta_information', 'PAM_PCWP_timestamp_in_TBME.json'), 'r') as f:
    data = json.load(f)
    if record_name in data:
      time_str = data[record_name]['PA_VI']['datetime'].strip()
      time_obj = datetime.strptime(time_str, '%H:%M:%S')
      return time_obj
    else:
      raise IndexError(f'Record {record_name} not in PAM_PCWP_timestamp_in_TBME.json')


def get_guessed_challenge_time(path, record_name):
  """
  Some challenges were not logged in PAM_PCWP_timestamp_in_TBME.json, so it is
  difficult to know exactly when to start recording for physiologic challenge.
  Empiricall set as the halfway between the last chamber time and overall end
  recording time.

  The following recordings included a physiologic challenge but were not included
  in PAM_PCWP_timestamp_in_TBME.json:
  
  TRM170.RHC2
  TRM176.RHC1
  TRM199.RHC1
  TRM203.RHC1
  TRM222.RHC1
  TRM253.RHC1
  TRM259.RHC2
  TRM274.RHC1
  TRM278.RHC1
  TRM279.RHC1

  The TRM325-RHC1 entry in PAM_PCWP_timestamp_in_TBME.json did not have a valid
  PA_VI value.

  Args:
    path (str): SCG-RHC database path.
    record_name (str): Record name.

  Returns:
    challenge_time (datetime): Time without date of when effects of physiologic
    challenge were presumably in full effect.
  """
  last_chamber_time = get_last_chamber_time(path, record_name) 
  _ , stop_time = get_start_and_stop_times(path, record_name)
  challenge_time = stop_time - ((stop_time - last_chamber_time) / 2)
  return challenge_time


def get_challenge_time(path, record_name, start_time):
  """
  Args:
    path (str): SCG-RHC database path.
    record_name (str): Record name.
    start_time (datetime): Match challenge time day, month, and year to start time.

  Returns:
    challenge_time (datetime): Time without date of when effects of physiologic
    challenge were presumably in full effect. Tries to pull time from 
    PAM_PCWP_timestamp_in_TBME.json if possible; otherwise, estimates time
    from midway between last chamber time and overall end recording time.
  """
  try:
    challenge_time = get_logged_challenge_time(path, record_name)
  except (IndexError, ValueError):
    challenge_time = get_guessed_challenge_time(path, record_name)
  challenge_time = challenge_time.replace(
    year=start_time.year, month=start_time.month, day=start_time.day)
  return challenge_time
  

def get_value_segments_from_record(
  db_path, record_name, acc_channels, ecg_channels, RHC_df, RHC_col, 
  segment_size, sample_rate):
  """
  Args:
    db_path (str): SCG-RHC database path.
    record_name (str): Record name.
    acc_channels (list[str]): ACC channels to include.
    ecg_channels (list[str]): ECG channels to include.
    RHC_df (DataFrame): Contains various RHC values for all recordings.
    RHC_col (str): RHC value of interest.
    segment_size (int): Number of samples in segment.
    sample_rate (int): Sample rate.

  Returns:
    segments (list[dict]): Segments with patch signals, patient
    attributes, and RHC values.
  """
  
  # Subset RHC df for the given record.
  subdf = RHC_df[RHC_df['Study ID'].str.strip() == record_name.replace('-', '.')]
  
  # Get start and stop times for recording.
  start_time, stop_time = get_start_and_stop_times(db_path, record_name)
  
  # Get static patient attributes.
  with open(os.path.join(db_path, 'processed_data', f'{record_name}.json'), 'r') as f:
    data = json.load(f)
    height = data['height']
    weight = data['weight']
    bmi = height / weight / weight * 10_000
    is_challenge = True if data['IsChallenge'] == 1 else False

  # Add RHC vals.
  RHC_vals = [] 
  
  # Add baseline RHC vals.
  baseline_RHC = subdf[subdf['RHC Phase'] == 'Baseline']
  if baseline_RHC.shape[0] > 1:
    raise ValueError(f'Multiple baselines found for record {record_name}')
  baseline_RHC = baseline_RHC.iloc[0][RHC_col]
  RHC_vals.append(baseline_RHC)
  
  # Add challenge RHC val, if exists, only has 1 post-challenge interval, and 
  # not TRM176.RHC1 since the patient underwent exercise challenge, which likely 
  # affected accelerometer readings.
  challenge_RHC = subdf[subdf['RHC Phase'] != 'Baseline']
  if challenge_RHC.shape[0] == 1:
    challenge_RHC = challenge_RHC.iloc[0][RHC_col]
    RHC_vals.append(challenge_RHC)

  # Add pre- and post-challenge intervals.
  challenge_intervals = []
  if not is_challenge:
    challenge_intervals.append((0, -1))
  else:
    challenge_time = get_challenge_time(db_path, record_name, start_time)
    challenge_index = get_sample_index(start_time, challenge_time, sample_rate)
    challenge_intervals.append((0, challenge_index))
    challenge_intervals.append((challenge_index, -1))
  
  # Add segments.
  segments = []
  for interval, RHC in zip(challenge_intervals, RHC_vals):
    if RHC.isnumeric():
      RHC = float(RHC)
    else:
      continue
    start, stop = interval
    acc_signal = get_channels(db_path, record_name, acc_channels, start, stop)
    ecg_signal = get_channels(db_path, record_name, ecg_channels, start, stop)
    num_segments = acc_signal.shape[0] // segment_size
    for i in range(num_segments):
      start = i * segment_size
      stop = start + segment_size
      acc_segment = acc_signal[start:stop]
      ecg_segment = ecg_signal[start:stop]
      segment = {
        'acc': acc_segment,
        'bmi': bmi,
        'rhc': RHC,
        'record_name': record_name,
        'start': start,
        'stop': stop
      }
      if ecg_channels:
        segment['ecg'] = ecg_segment
      segments.append(segment)
  return segments


def get_value_segments(
  db_path, record_paths, acc_channels, ecg_channels, RHC_col, RHC_df, 
  segment_size, sample_rate):
  """
  Args:
    db_path (str): SCG-RHC database path.
    record_paths (list[str]): Record paths.
    acc_channels (list[str]): ACC channels to include.
    ecg_channels (list[str]): ECG channels to include.
    RHC_col (str): RHC value of interest.
    RHC_df (DataFrame): Contains various RHC values for all recordings.
    segment_size (int): Number of samples in segment.
    sample_rate (int): Sample rate.

  Returns:
    segments (list[Segment]): Segments with patch signals, patient
    attributes, and RHC value collected from given list of record paths.
  """
  segments = []
  for record_path in record_paths:
    segments.extend(get_value_segments_from_record(
      db_path, record_path, acc_channels, ecg_channels, RHC_df, RHC_col, 
      segment_size, sample_rate))
  return segments


def get_chamber_intervals(db_path, record_name, chamber, sample_rate):
  """
  Args:
    db_path (str): SCG-RHC database path.
    record_name (str): Record name.
    chamber (str): Chamber name.
    sample_rate (int): Sample rate (Hz).

  Returns:
    intervals (list[tuple]): Start and stop index of when cath was in the given
    chamber.
  """
  intervals = []
  with open(os.path.join(db_path, 'processed_data', f'{record_name}.json'), 'r') as f:
    data = json.load(f)
    macStTime = datetime.strptime(data['MacStTime'].split()[1], '%H:%M:%S')
    macEndTime = datetime.strptime(data['MacEndTime'].split()[1], '%H:%M:%S')
    chamEvents = data['ChamEvents_in_s']
    if isinstance(chamEvents, dict):
      chamEvents['END'] = (macEndTime - macStTime).total_seconds()
      chamEvents = sorted(chamEvents.items(), key=lambda x: x[1])
      intervals = []
      for i, event in enumerate(chamEvents[:-1]):
        if event[0].split('_')[0] == chamber:
          intervals.append((int(event[1] * sample_rate), int(chamEvents[i+1][1] * sample_rate)))
  return intervals


def get_waveform_segments_from_record(
  db_path, record_name, acc_channels, ecg_channels, chamber, segment_size, 
  sample_rate, flat_threshold, flat_min_duration, straight_threshold, min_RHC):
  """
  Args:
    db_path (str): SCG-RHC database path.
    record_name (str): Record name.
    acc_channels (list[str]): List of ACC channels to include
    ecg_channels (list[str]): List of ECG channels to include.
    chamber (str): Chamber waveform to predict.
    segment_size (int): Number of samples in segment.
    sample_rate (int): Sample rate (Hz).
    flat_threshold (float): Amplitude threshold to be considered flat.
    flat_min_duration (float): Minimum duration (seconds) to be considered flat.
    straight_threshold (float): R squared threshold to be considered straight line.
    min_RHC (float): Minimum allowable RHC value.

  Notes:
    Do not include segments unless they contain all specified channels.

  Returns:
    segments (list[dict]): Segments with aim of predicting chamber waveform.
  """
  segments = []
  for interval in get_chamber_intervals(db_path, record_name, chamber, sample_rate):
    acc_signal = get_channels(db_path, record_name, acc_channels, interval[0], interval[1])
    ecg_signal = get_channels(db_path, record_name, ecg_channels, interval[0], interval[1])
    rhc_signal = get_channels(db_path, record_name, ['RHC_pressure'], interval[0], interval[1])
    num_segments = acc_signal.shape[0] // segment_size
    for i in range(num_segments):
      start = i * segment_size
      stop = start + segment_size
      acc_segment = acc_signal[start:stop]
      ecg_segment = ecg_signal[start:stop]
      rhc_segment = rhc_signal[start:stop]
      if not has_noise(
        rhc_segment[:, 0], flat_threshold, flat_min_duration, straight_threshold, 
        min_RHC, sample_rate):
        segment = {
          'acc': acc_segment,
          'rhc': rhc_segment,
          'record_name': record_name,
          'start': start,
          'stop': stop
        }
        if ecg_channels:
          segment['ecg'] = ecg_segment
        segments.append(segment)
  return segments


def get_waveform_segments(
  db_path, record_names, acc_channels, ecg_channels, chamber, segment_size,
  sample_rate, flat_threshold, flat_min_duration, straight_threshold, min_RHC):
  """
  Args:
    db_path (str): SCG-RHC database path.
    record_names (str): Record names.
    acc_channels (list[str]): List of ACC channels to include
    ecg_channels (list[str]): List of ECG channels to include.
    chamber (str): Chamber waveform to predict.
    segment_size (int): Number of samples in segment.
    sample_rate (int): Sample rate (Hz).
    flat_threshold (float): Amplitude threshold to be considered flat.
    flat_min_duration (float): Minimum duration (seconds) to be considered flat.
    straight_threshold (float): R squared threshold to be considered straight line.
    min_RHC (float): Minimum allowable RHC value.

  Notes:
    Do not include segments unless they contain all specified channels.

  Returns:
    segments (list[dict]): Segments with aim of predicting chamber waveform.
  """
  segments = []
  for record_name in record_names:
    segments.extend(get_waveform_segments_from_record(
      db_path, record_name, acc_channels, ecg_channels, chamber, segment_size,
      sample_rate, flat_threshold, flat_min_duration, straight_threshold, min_RHC))
  return segments


def get_groups(list_to_split, n):
  """
  Args:
    list_to_split (list): The list to split.
    n (int): Desired length of sublists.

  Returns:
    groups (list[list]): List of lists, where each inner length is a sublist of
    the original list with a length of n. List is shuffled before splitting. 
    Groups are created iteratively from left to right. 
  """
  groups = []
  random.shuffle(list_to_split)
  for i in range(0, len(list_to_split), n):
    group = list_to_split[i:i+n]
    if len(group) == n:
      groups.append(group)
  return groups


def save_dataset(
  db_path, dataset_dir_path, outcome_type, batch_size, acc_channels, ecg_channels, 
  RHC_col, chamber, num_tests, num_folds, segment_size, sample_rate, 
  flat_threshold, flat_min_duration, straight_threshold, min_RHC):
  """
  Save dataset with given parameters.

  Args:
    db_path (str): SCG-RHC database path.
    dataset_dir_path (str): Dataset directory path.
    outcome_type (str): Indicates type of prediction to perform, 'value' or 'waveform'.
    batch_size (int): Batch size.
    acc_channels (list[str]): ACC channels to include.
    ecg_channels (list[str]): ECG channels to include.
    RHC_col (str): RHC column of interest when performing value prediction.
    chamber (str): Heart chamber of interest when performing waveform prediction.
    num_test (int): Number of patients to include in a test set.
    num_folds (int): Number of folds to create.
    segment_size (float): Segment duration (sec).
    sample_rate (int): Sample rate (Hz).
    flat_threshold (float): Amplitude threshold to be considered flat.
    flat_min_duration (float): Minimum duration (seconds) to be considered flat.
    straight_threshold (float): R squared threshold to be considered straight line.
    min_RHC (float): Minimum allowable RHC value.
  """

  start_time = time()
  timelog(f'Create {dataset_dir_path}', start_time)

  # Load RHC dataframe.
  RHC_df = pd.read_csv(os.path.join(db_path, 'meta_information', 'RHC_values.csv'))

  # Convert segment size in sec to number of samples.
  segment_size = int(segment_size * sample_rate)

  # Check outcome type
  assert outcome_type == 'value' or outcome_type == 'waveform'

  # Get all patients and record names that may be included in a test set.
  if outcome_type == 'value':
    _, all_test_records = get_value_test_codes_and_record_names(db_path)
  elif outcome_type == 'waveform':
    _, all_test_records = get_waveform_test_codes_and_record_names(db_path)

  # Randomly split test records into groups of a given length.
  test_record_groups = get_groups(all_test_records, num_tests)[:num_folds]

  # Create different hold out datasets.
  for i, test_records in enumerate(test_record_groups):

    timelog(f'Create fold {i+1}/{len(test_record_groups)}', start_time)
    
    # Check if train loader already exists.
    train_loader_path = os.path.join(dataset_dir_path, f'{i+1}_train_loader.pkl')
    if os.path.exists(train_loader_path):
      raise FileExistsError(f'File {train_loader_path} already exists!')

    # Check if valid loader already exists.
    valid_loader_path = os.path.join(dataset_dir_path, f'{i+1}_valid_loader.pkl')
    if os.path.exists(valid_loader_path):
      raise FileExistsError(f'File {valid_loader_path} already exists!')

    # Check if test loader already exists.
    test_loader_path = os.path.join(dataset_dir_path, f'{i+1}_test_loader.pkl')
    if os.path.exists(test_loader_path):
      raise FileExistsError(f'File {test_loader_path} already exists!')
    
    # Get test patient codes.
    test_patients = [get_patient_code(i) for i in test_records]

    # Get train patients and record names.
    train_patients, train_records = get_train_codes_and_record_names(db_path, test_records)

    # Random state for spltiting train and valid segments.
    random_state = random.randint(1, 1000)

    # Get segments for predicting a certain value if indicated.
    if outcome_type == 'value':
      
      test_segs = get_value_segments(
        db_path, test_records, acc_channels, ecg_channels, 
        RHC_col, RHC_df, segment_size, sample_rate)
      
      nontest_segs = get_value_segments(
        db_path, train_records, acc_channels, ecg_channels, 
        RHC_col, RHC_df, segment_size, sample_rate)
    
    # Get segments for predicting a certain waveform if indicated.
    if outcome_type == 'waveform':
      
      test_segs = get_waveform_segments(
        db_path, test_records, acc_channels, ecg_channels, chamber, 
        segment_size, sample_rate, flat_threshold, flat_min_duration, 
        straight_threshold, min_RHC)
      
      nontest_segs = get_waveform_segments(
        db_path, train_records, acc_channels, ecg_channels, chamber, 
        segment_size, sample_rate, flat_threshold, flat_min_duration, 
        straight_threshold, min_RHC)
    
    train_segs, valid_segs = train_test_split(nontest_segs, train_size=0.9, random_state=random_state)

    # Get datasets.
    train_set = SCGDataset(train_segs, segment_size)
    valid_set = SCGDataset(valid_segs, segment_size)
    test_set = SCGDataset(test_segs, segment_size)

    # Get data loaders.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    # Save train data loader.
    with open(train_loader_path, 'wb') as f:
      pickle.dump(train_loader, f)

    # Save valid data loader.
    with open(valid_loader_path, 'wb') as f:
      pickle.dump(valid_loader, f)

    # Save test data loader.
    with open(test_loader_path, 'wb') as f:
      pickle.dump(test_loader, f)

    # Save record log.
    record_log_path = os.path.join(dataset_dir_path, f'{i+1}_record_log.json')
    with open(record_log_path, 'w') as f:
      f.write(
        json.dumps({
          'test_patients': test_patients,
          'test_records': test_records,
          'train_patients': train_patients,
          'train_records': train_records,
          'random_state': random_state,
          'train segments': len(train_segs),
          'valid segments': len(valid_segs),
          'test_segments': len(test_segs)
        }, indent=4)
      )


def run(dir_path):
  """
  Args:
    dir_path (str): Project directory path.
  """
  filepath = os.path.join(dir_path, 'params.json')
  with open(filepath, 'r') as f:
    data = json.load(f)
    save_dataset(
      DATABASE_PATH,
      dir_path,
      data['outcome_type'],
      data['batch_size'],
      data['acc_channels'],
      data['ecg_channels'],
      data['RHC_col'],
      data['chamber'],
      data['num_tests'],
      data['num_folds'],
      data['segment_size'],
      SAMPLE_RATE,
      data['flat_threshold'],
      data['flat_min_duration'],
      data['straight_threshold'],
      data['min_RHC']
    )
 

if __name__ == '__main__':
  dir_path = sys.argv[1]
  run(dir_path)

