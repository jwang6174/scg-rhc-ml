import json
import numpy as np
import os
import pickle
import random
import sys
import wfdb
from datetime import datetime
from pathlib import Path
from pathutil import DATABASE_PATH
from sklearn.model_selection import train_test_split
from noiseutil import has_noise

# Waveform sample rate.
SAMPLE_RATE = 500


def get_record_names(db_path):
  """
  Args:
    db_path (str): SCG-RHC database path.

  Returns:
    names (list[str]): Record names.
  """
  names = set()
  for filename in os.listdir(os.path.join(db_path, 'processed_data')):
    if filename.endswith('.dat') or filename.endswith('.hea'):
      names.add(Path(filename).stem)
  names = sorted(list(names))
  return names


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


def get_PA_challenge_interval(record_name, macStTime, db_path, sample_rate):
  """
  Args:
    record_name (str): Record name.
    macStTime (datetime): Start time of whole recording.
    db_path (str): SCG-RHC database.

  Notes:
    If no challenge administered or time was not recorded, then return None.

  Returns:
    start_idx (int): Sample index when catheter was in the pulmonary artery
    during vasodilator infusion if administered and times were recorded in
    PAM_PCWP_timestamp_in_TBME.json. 

    end_idx (int): Sample index when catheter was in wedge position during
    vasodilator infusion, essentially marking the end of the PA tracing.
  """
  filepath = os.path.join(db_path, 'meta_information', 'PAM_PCWP_timestamp_in_TBME.json')
  with open(filepath, 'r') as f:
    data = json.load(f)
    if record_name in data:
      try:
        start_str = data[record_name]['PA_VI']['datetime'].strip()
        start_obj = datetime.strptime(start_str, '%H:%M:%S')
        start_idx = get_sample_index(macStTime, start_obj, sample_rate)
        end_str = data[record_name]['PCW_VI']['datetime'].strip()
        end_obj = datetime.strptime(end_str, '%H:%M:%S')
        end_idx = get_sample_index(macStTime, end_obj, sample_rate)
        return start_idx, end_idx
      except:
        return None


def get_chamber_intervals(record_name, chamber, db_path, sample_rate):
  """
  Args:
    record_name (str): Record name.
    chamber (str): Chamber name.
    db_path (str): SCG-RHC database path.
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
      for i, event in enumerate(chamEvents[:-1]):
        if event[0].split('_')[0] == chamber:
          intervals.append((int(event[1] * sample_rate), 
                            int(chamEvents[i+1][1] * sample_rate)))
    
  if chamber == 'PA':
    PA_challenge_interval = get_PA_challenge_interval(record_name, macStTime, 
                                                      db_path, sample_rate)
    if PA_challenge_interval is not None:
      intervals.append(PA_challenge_interval)

  return intervals


def get_channels(record_name, channel_names, start, stop, db_path):
  """
  Args:
    record_name (str): Record name.
    channel_names (list[str]): Channel names.
    start (int): Start index for record segment.
    stop (int): Stop index for record segment.
    db_path (str): SCG-RHC databse path.

  Raises:
    IndexError: If channel name cannot be found in record.

  Returns:
    channels (ndarray): An (MxN) 2d numpy array, where M is the signal length and
    N is the number of channels.
  """     
  record = wfdb.rdrecord(os.path.join(db_path, 'processed_data', record_name))                              
  indexes = [record.sig_name.index(name) for name in channel_names]                                
  channels = record.p_signal[start:stop, indexes]
  return channels


def get_patient_code(record_name):
  """
  Args:
    record_name (str): Record name that includes patient code.

  Returns:
    code (str): Patient code.
  """
  code = record_name.split('-')[0]
  return code


def get_record_names_with_challenge(db_path):
  """
  Args:
    db_path (str): SCG-RHC database path.

  Returns:
    with_challenge (list[str]): Record names with physiologic challenge 
    (eg, nitro, dobutamine, fluids, leg raise, etc).
  """
  with_challenge = []
  for name in get_record_names(db_path):
    with open(os.path.join(db_path, 'processed_data', f'{name}.json'), 'r') as f:
      data = json.load(f)
      if data['IsChallenge'] == 1:
        with_challenge.append(name)
  return with_challenge



def get_record_names_without_challenge(db_path):
  """
  Args:
    db_path (str): SCG-RHC database path.

  Returns:
    no_challenge (list[str]): Record names without physiologic challenge.
  """
  all_names = get_record_names(db_path)
  with_challenge = get_record_names_with_challenge(db_path)
  no_challenge = sorted(list(set(all_names) - set(with_challenge)))
  return no_challenge


def get_record_names_of_multiple_caths(db_path):
  """
  Args:
    db_path (str): SCG-RHC database path.

  Returns:
    multi (list[str]): Record names of multiple caths.
  """
  multi = []
  counts = {}
  record_names = get_record_names(db_path)
  
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


def get_record_names_of_single_caths(db_path):
  """
  Args:
    db_path (str): SCG-RHC database path.

  Returns:
    single (list[str]): Record names of only one cath.
  """
  single = []
  all_names = get_record_names(db_path)
  multi_names = get_record_names_of_multiple_caths(db_path)
  single = sorted(list(set(all_names) - set(multi_names)))
  return single


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


def get_global_stats(segments, signal_names):
  """
  Calculate some stats for a specific signal type, including mean, standard 
  deviation, min, and max.

  Args:
    segments (list[dict]): Segments of interest.
    signal_names (list[str]): Names of signals of interest.
  
  Returns:
    stats (dict): Different stats for each signal type.
  """
  joined = {}
  for segment in segments:
    for signal_name in signal_names:
      signal = segment.get(signal_name)
      if signal is not None:
        joined[signal_name] = signal
      else:
        joined[signal_name] = np.hstack((joined[signal_name], signal))

  stats = {}
  for signal_name, joined_segment in joined.items():
    stats[f'{signal_name}_avg'] = np.mean(joined_segment)
    stats[f'{signal_name}_std'] = np.std(joined_segment)
    stats[f'{signal_name}_max'] = np.max(joined_segment)
    stats[f'{signal_name}_min'] = np.min(joined_segment)

  return stats


def add_local_stats(segments, signal_names):
  """
  Calculate local stats include min and max for all segments.

  Args:
    segments (list[dict]): All segments.
    signal_names (list[str]): Signal names to calculate local stats for.
  """
  for segment in segments:
    for name in signal_names:
      signal = segment[name]
      min = np.min(signal)
      max = np.max(signal)
      segment[f'{name}_min'] = min
      segment[f'{name}_max'] = max


def get_test_record_names(db_path):
  """
  Identify record names without challenge and with patients who underwent
  only a single catheterization. These are potential records that may be used
  in a test set. The goal is for the model to evaluate tracings from patients
  it has not seen before. If challenge and multiple catheterizations were
  included, then then would be the possibility of a patient being included
  in both training and test sets.

  Args:
    path (str): SCG-RHC database path.

  Returns:
    record_names (list[str]): Record names in test set.
  """
  set1 = set(get_record_names_without_challenge(db_path))
  set2 = set(get_record_names_of_single_caths(db_path))
  record_names = list(set1.intersection(set2))
  return record_names


def get_train_record_names(test_record_names, db_path):
  """
  Get patient codes and record names to be included in a train set by subtracting
  test record names from all record names.

  Args:
    test_record_names (list[str]):  Record names in test set.
    db_path (str): SCG-RHC database path.

  Returns:
    record_names (list[str]): Record names in train set.
  """
  record_names = sorted(list(set(get_record_names(db_path)) - set(test_record_names)))
  return record_names


def get_record_segments(record_name, acc_channels, chamber, segment_size, 
                        flat_amp_threshold, flat_min_duration, straight_threshold, 
                        min_RHC, max_RHC, db_path, sample_rate):
  """
  Args:
    record_name (str): Record name.
    acc_channels (list[str]): List of ACC channels to include
    ecg_channels (list[str]): List of ECG channels to include.
    chamber (str): Chamber waveform to predict.
    segment_size (float): Segment duration (s).
    flat_amp_threshold (float): Amplitude threshold to be considered flat.
    flat_min_duration (float): Minimum duration (seconds) to be considered flat.
    straight_threshold (float): R squared threshold to be considered straight line.
    min_RHC (float): Minimum allowable RHC value.
    max_RHC (float): Maximum allowable RHC value.
    db_path (str): SCG-RHC database path.
    sample_rate (int): Sample rate (Hz).

  Returns:
    record_segments (list[dict]): Record segments.
  """
  record_segments = []
  segment_size *= sample_rate
  for chamber_start, chamber_end in get_chamber_intervals(record_name, chamber, 
                                                          db_path, sample_rate):
    acc = get_channels(record_name, acc_channels, chamber_start, chamber_end, db_path)
    rhc = get_channels(record_name, ['RHC_pressure'], chamber_start, chamber_end, db_path)
    num_segments = int(acc.shape[0] // segment_size)
    for i in range(num_segments):
      start_idx = int(i * segment_size)
      end_idx = int(start_idx + segment_size)
      acc_seg = acc[start_idx:end_idx]
      rhc_seg = rhc[start_idx:end_idx]
      if not has_noise(rhc_seg[:, 0], flat_amp_threshold, flat_min_duration, 
                       straight_threshold, min_RHC, max_RHC, sample_rate):
        segment = {
          'acc': acc_seg,
          'rhc': rhc_seg,
          'record_name': record_name,
          'start_idx': start_idx,
          'end_idx': end_idx,
        }
        record_segments.append(segment)
  return record_segments


def get_dataset_segments(record_names, acc_channels, chamber, segment_size, 
                         flat_amp_threshold, flat_min_duration, straight_threshold, 
                         min_RHC, max_RHC, db_path, sample_rate):
  """
  Args:
    record_names (list[str]): Names of records to get segments from.
    acc_channels (list[str]): List of ACC channels to include
    chamber (str): Chamber waveform to predict.
    segment_size (int): Number of samples in segment.
    flat_amp_threshold (float): Amplitude threshold to be considered flat.
    flat_min_duration (float): Minimum duration (seconds) to be considered flat.
    straight_threshold (float): R squared threshold to be considered straight line.
    min_RHC (float): Minimum allowable RHC value.
    max_RHC (float): Maximum allowable RHC value.
    db_path (str): SCG-RHC database path.
    sample_rate (int): Sample rate (Hz).

  Returns:
    dataset_segments (list[dict]): Dataset segments.
  """
  dataset_segments = []
  for record_name in record_names:
    record_segments = get_record_segments(record_name, acc_channels, chamber, 
                                          segment_size, flat_amp_threshold, 
                                          flat_min_duration, straight_threshold, 
                                          min_RHC, max_RHC, db_path, sample_rate)
    dataset_segments.extend(record_segments)
  return dataset_segments


def save_dataset(dataset_name, dataset_mirror, acc_channels, chamber, segment_size,
                 num_tests, num_folds, flat_amp_threshold, flat_min_duration, 
                 straight_threshold, min_RHC, max_RHC, db_path, sample_rate):
  """
  Save list of train, validation, and test segments as pickle files.

  Args:
    dataset_name (str): Dataset name.
    dataset_mirror (str): If set, new dataset will mirror this dataset in terms
      of folds, test records, and train records.
    batch_size (int): Batch size.
    acc_channels (list[str]): ACC channels.
    chamber (str): Heart chamber of interest when performing waveform prediction.
    segment_size (float): Segment duration (sec).
    num_tests (int): Number of records in test set.
    num_folds (int): Number of test folds.
    flat_amp_threshold (float): Amplitude threshold to be considered flat.
    flat_min_duration (float): Minimum duration (seconds) to be considered flat.
    straight_threshold (float): R squared threshold to be considered straight line.
    min_RHC (float): Minimum allowable RHC value.
    max_RHC (float): Maximum allowable RHC value.
    db_path (str): SCG-RHC database path.
    sample_rate (int): Sample rate (Hz).
  """
  print(f'Run recordutil.py for {dataset_name}')

  # Define signal names.
  signal_names = ['acc', 'rhc']

  # Use this if creating a new datset from scratch.
  if dataset_mirror is None:

    # Get names of all records that may be included in a test set.
    all_test_records = get_test_record_names(db_path)

    # Randomly split test records into groups of a given length.
    test_record_groups = get_groups(all_test_records, num_tests)[:num_folds]

  # Use this if mirroring a prior dataset.
  else:
    test_record_groups = []
    dataset_filepath = os.path.join('datasets', dataset_mirror)
    for filename in sorted(os.listdir(dataset_filepath)):
      if filename[:12] == 'segment_info':
        with open(os.path.join(dataset_filepath, filename), 'r') as f:
          data = json.load(f)
          test_record_groups.append(data['test_records'])


  # Create different hold out sets.
  for i, test_records in enumerate(test_record_groups):
    print(f'Create fold {i+1}/{num_folds}')

    # Identify training records by subtracting test records from all records.
    train_records = get_train_record_names(test_records, db_path)

    # Get training segments.
    train_segments = get_dataset_segments(train_records, acc_channels, chamber, 
                                          segment_size, flat_amp_threshold, 
                                          flat_min_duration, straight_threshold, 
                                          min_RHC, max_RHC, db_path, sample_rate)

    # Get test segments.
    test_segments = get_dataset_segments(test_records, acc_channels, chamber, 
                                         segment_size, flat_amp_threshold, 
                                         flat_min_duration, straight_threshold, 
                                         min_RHC, max_RHC, db_path, sample_rate)

    # Calculate global stats for each signal across all segments and save 
    # to file. To be used for feature normalization during training.
    all_segments = train_segments + test_segments
    global_stats = get_global_stats(all_segments, signal_names)
    global_stats_path = os.path.join('datasets', dataset_name, f'global_stats_{i+1}.json')
    with open(global_stats_path, 'w') as f:
      json.dump(global_stats, f, indent=2)

    # Add local stats to all segments.
    add_local_stats(all_segments, signal_names)

    # Subset random portion of training segments for validation.
    train_segments, valid_segments = train_test_split(train_segments,
                                                      train_size=0.95,
                                                      shuffle=True)
  
    # Save train segments.
    train_path = os.path.join('datasets', dataset_name, f'train_segments_{i+1}.pkl')
    with open(train_path, 'wb') as f:
      pickle.dump(train_segments, f)

    # Save valid segments.
    valid_path = os.path.join('datasets', dataset_name, f'valid_segments_{i+1}.pkl')
    with open(valid_path, 'wb') as f:
      pickle.dump(valid_segments, f)

    # Save test segments.
    test_path = os.path.join('datasets', dataset_name, f'test_segments_{i+1}.pkl')
    with open(test_path, 'wb') as f:
      pickle.dump(test_segments, f)

    # Save segment info.
    log_path = os.path.join('datasets', dataset_name, f'segment_info_{i+1}.json')
    with open(log_path, 'w') as f:
      json_str = json.dumps({
        'train_records': train_records,
        'test_records': test_records,
        'train_segments': len(train_segments),
        'valid_segments': len(valid_segments),
        'test_segments': len(test_segments),
      }, indent=4)
      f.write(json_str)


def run(dataset_name):
  """
  Run recordutil.py for given dataset.

  Args:
    dataset_name (str): Dataset name.
  """
  dataset_params_path = os.path.join('datasets', dataset_name, 'params.json')
  with open(dataset_params_path, 'r') as f:
    params = json.load(f)
    save_dataset(
      dataset_name,
      params['dataset_mirror'],
      params['acc_channels'],
      params['chamber'],
      params['segment_size'],
      params['num_tests'],
      params['num_folds'],
      params['flat_amp_threshold'],
      params['flat_min_duration'],
      params['straight_threshold'],
      params['min_RHC'],
      params['max_RHC'],
      DATABASE_PATH,
      SAMPLE_RATE
    )


if __name__ == '__main__':
  dataset_name = sys.argv[1]
  run(dataset_name)
