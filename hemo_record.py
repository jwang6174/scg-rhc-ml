import json
import numpy as np
import os
import pandas as pd
import random
import sys
from pathutil import DATABASE_PATH
from wave_record import *

# Define number of time steps.
NUM_STEPS = 10 * SAMPLE_RATE

# Define RHC types of interest. Need to strongly consider the distribution of
# each type, as some records may be need to be duplicated a certain number
# of times to maintain data balance for one RHC type, but a separate number
# of times for a different RHC type.
RHC_TYPE = 'Avg. COmL/min'

# Define ACC channels to use.
ACC_CHANNELS = [
  'patch_ACC_lat',
  'patch_ACC_dv',
  'patch_ACC_hf'
]

# Define ECG channels to use.
ECG_CHANNELS = [
  'patch_ECG'
]

# Segment info to ignore when normalizing.
IGNORE = set(['start', 'end', 'record_name'])

# Define number of test records and folds to be used.
NUM_VALID = 7
NUM_TESTS = 7
NUM_FOLDS = 3


def get_start_and_end_times(record_name):
  """
  Args:
    record_name (str): Record name.

  Returns:
    start (datetime): Recording start time without date.
    end (datetime): Recording end time without date.
  """
  path = os.path.join(DATABASE_PATH, 'processed_data', f'{record_name}.json')
  with open(path, 'r') as f:
    data = json.load(f)
    start = datetime.strptime(data['MacStTime'], '%d-%b-%Y %H:%M:%S')
    end = datetime.strptime(data['MacEndTime'], '%d-%b-%Y %H:%M:%S')
    return start, end


def get_logged_challenge_time(record_name):
  """
  Get start of recording after physiologic challenge presumably in full effect,
  as logged in PAM_PCWP_timestamp_in_TBME.json, assuming that PA_VI represents
  the start.

  Args:
    record_name (str): Record name.

  Raises:
    IndexError: If record name not in PAM_PCWP_timestamp_in_TBME.json, suggesting
    that no challenge was recorded in the given record.

  Returns:
    time_obj (datetime): Time without date of when effects of physiologic
    challenge were presumably in full effect.
  """
  filename = 'PAM_PCWP_timestamp_in_TBME.json'
  filepath = os.path.join(os.path.join(DATABASE_PATH, 'meta_information', filename))
  with open(filepath, 'r') as f:
    data = json.load(f)
    if record_name in data:
      time_str = data[record_name]['PA_VI']['datetime'].strip()
      time_obj = datetime.strptime(time_str, '%H:%M:%S')
      return time_obj
    else:
      raise IndexError(f'Record {record_name} not in {filename}')


def get_last_chamber_time(record_name):
  """
  Args:
    record_name (str): Record name.

  Returns:
    time_obj (datetime) Last chamber time without date.
  """
  path = os.path.join(DATABASE_PATH, 'processed_data', f'{record_name}.json')
  with open(path, 'r') as f:
    data = json.load(f)
    last_time = None
    for chamber, time_str in data['ChamEvents'].items():
      time_obj = datetime.strptime(time_str.strip(), '%d-%b-%Y %H:%M:%S %p')
      if last_time is None or time_obj > last_time:
        last_time = time_obj
    return last_time


def get_guessed_challenge_time(record_name):
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
    record_name (str): Record name.

  Returns:
    challenge_time (datetime): Time without date of when effects of physiologic
    challenge were presumably in full effect.
  """
  last_chamber_time = get_last_chamber_time(record_name) 
  _ , end_time = get_start_and_end_times(record_name)
  challenge_time = end_time - ((end_time - last_chamber_time) / 2)
  return challenge_time


def get_challenge_time(record_name, start_time):
  """
  Args:
    record_name (str): Record name.
    start_time (datetime): Match challenge time day, month, and year to start time.

  Returns:
    challenge_time (datetime): Time without date of when effects of physiologic
    challenge were presumably in full effect. Tries to pull time from 
    PAM_PCWP_timestamp_in_TBME.json if possible; otherwise, estimates time
    from midway between last chamber time and overall end recording time.
  """
  try:
    challenge_time = get_logged_challenge_time(record_name)
  except (IndexError, ValueError):
    challenge_time = get_guessed_challenge_time(record_name)
    challenge_time = challenge_time.replace(year=start_time.year, 
                                            month=start_time.month,
                                            day=start_time.day)
  return challenge_time


def get_sample_index(start_time, sample_time):
  """
  Args:
    start_time (datetime): Start time without date.
    sample_time (datetime): Sample time without date.

  Returns:
    sample_index (int): Sample index corresponding to given sample time.
  """
  start_time = start_time.replace(year=2000, month=1, day=1)
  sample_time = sample_time.replace(year=2000, month=1, day=1)
  diff = (sample_time - start_time).total_seconds()
  sample_index = int(diff * SAMPLE_RATE)
  return sample_index


def add_record_segments(dataset_segments, record_name, rhc_df):
  """
  Add record segments to list of dataset segments.

  Args:
    dataset_segments (list[dict]): List of dataset segments.
    record_name (str): Name of record to get segments from.
    rhc_df (DataFrame): Contains different RHC values for all records.
  """

  # Subset RHC dataframe for the given record.
  rhc_subdf = rhc_df[rhc_df['Study ID'].str.strip() == record_name.replace('-', '.')]

  # Get record start and end times.
  start_time, end_time = get_start_and_end_times(record_name)

  # Get static patient attributes.
  metapath = os.path.join(DATABASE_PATH, 'processed_data', f'{record_name}.json')
  with open(metapath, 'r') as f:
    data = json.load(f)
    height = data['height']
    weight = data['weight']
    is_challenge = bool(data['IsChallenge'])

  # Load WFDB record.
  record = wfdb.rdrecord(os.path.join(DATABASE_PATH, 'processed_data', record_name))

  # Get baseline RHC val
  baseline_df = rhc_subdf[rhc_subdf['RHC Phase'] == 'Baseline']
  if baseline_df.shape[0] > 1:
    raise ValueError(f'Multiple baselines found for record {record_name}')
  baseline_rhc_val = baseline_df.iloc[0][RHC_TYPE]
  all_rhc_vals = [baseline_rhc_val]

  # Add challenge RHC val, if exists and only has 1 post-challenge interval.
  challenge_rhc_vals = {}
  challenge_df = rhc_subdf[rhc_subdf['RHC Phase'] != 'Baseline']
  if challenge_df.shape[0] == 1:
    challenge_rhc_val = challenge_df.iloc[0][RHC_TYPE]
    all_rhc_vals.append(challenge_rhc_val)

  # Get ACC signals.
  acc_indexes = [record.sig_name.index(x) for x in ACC_CHANNELS]
  acc = record.p_signal[:, acc_indexes]

  # Get ECG signals.
  ecg_indexes = [record.sig_name.index(x) for x in ECG_CHANNELS]
  ecg = record.p_signal[:, ecg_indexes]

  # Get pre- and post-challenge intervals.
  challenge_intervals = []
  if not is_challenge:
    challenge_intervals.append((0, len(acc)))
  else:
    challenge_time = get_challenge_time(record_name, start_time)
    challenge_index = get_sample_index(start_time, challenge_time)
    challenge_intervals.append((0, challenge_index))
    challenge_intervals.append((challenge_index, len(acc)))

  # Add segments.
  segments = []

  # Iterate over baseline and challenges in record, if challenge exists.
  for (challenge_start, challenge_end), rhc_val in zip(challenge_intervals, all_rhc_vals):
    try:
      rhc_val = float(rhc_val)
    except:
      continue
    segment_start = challenge_start
    while segment_start < challenge_end:
      segment_end = int(segment_start + NUM_STEPS)
      acc_segment = acc[segment_start:segment_end,]
      ecg_segment = ecg[segment_start:segment_end,]
      if len(acc_segment) > 0 and len(ecg_segment) > 0:
        segment = {
          'acc': acc_segment,
          'ecg': ecg_segment,
          'height': np.array([height]),
          'weight': np.array([weight]),
          'bmi': np.array([height / weight / weight * 10_000]),
          'record_name': record_name,
          'start': segment_start,
          'end': segment_end,
          RHC_TYPE: np.array([rhc_val])
        }
        dataset_segments.append(segment)
      segment_start += NUM_STEPS // 4


def get_bin_info(segments):
  """
  Use Freedman-Diaconis rule for binning RHC values.

  Args:
    segments (list[dict]): Segments.

  Returns:
    bin_width (int): Bin width.
    num_bins (int): Number of bins.
    min_val (float): Minimum RHC value.
    max_val (float): Maximum RHC value.
  """
  data = [segment[RHC_TYPE] for segment in segments]
  iqr = np.percentile(data, 75) - np.percentile(data, 25)
  n = len(data)
  bin_width = int(np.ceil(2 * iqr / n ** (1/3)))
  min_val = int(np.floor(np.min(data)))
  max_val = int(np.ceil(np.max(data)))
  num_bins = int(np.ceil((max_val - min_val) / bin_width))
  return bin_width, num_bins, min_val, max_val


def balance_segments(segments):
  """
  Modify list of segments in place to be balanced using the Freedman-Diaconis
  rule for binning data.

  Args:
    segments (list[dict]): Segments.
  """

  # Get bin info.
  bin_width, num_bins, min_val, max_val = get_bin_info(segments)

  # Initialize bins.
  bins = {}
  for i in range(int(np.floor(min_val)), int(np.ceil(max_val)), bin_width):
    bins[i] = []

  # Add segments to bins.
  for segment in segments:
    for bin_start in bins.keys():
      if bin_start <= segment[RHC_TYPE] < bin_start + bin_width:
        bins[bin_start].append(segment)

  # Determine most frequent bin.
  most_freq_bin = None
  most_freq_cnt = 0
  for bin_start, bin_segments in bins.items():
    bin_cnt = len(bin_segments)
    if bin_cnt > most_freq_cnt:
      most_freq_bin = bin_start
      most_freq_cnt = bin_cnt

  # Balance segments.
  for bin_start, bin_segments in bins.items():
    orig_segments = list(bin_segments)
    if len(orig_segments) > 0:
      mod = most_freq_cnt / len(bin_segments)
      while mod >= 2:
        segments.extend(orig_segments)
        mod -= 1
      if mod > 1:
        prob = mod - 1
        sampled = random.sample(orig_segments, int(prob * len(orig_segments)))
        segments.extend(sampled)


def get_dataset_segments(record_names, rhc_df):
  """
  Args:
    record_names (list[str]): Names of records to get segments from.
    rhc_df (DataFrame): Contains different RHC values for all records.

  Returns:
    dataset_segments (list[dict]): Dataset segments.
  """
  dataset_segments = []
  for i, record_name in enumerate(record_names):
    add_record_segments(dataset_segments, record_name, rhc_df)
  balance_segments(dataset_segments)
  return dataset_segments


def get_global_stats(train_segments, valid_segments, test_segments):
  """
  Args:
    train_segments (list[dict]): Train segments.
    valid_segments (list[dict]): Valid segments.
    test_segments (list[dict]): Test segments.

  Returns:
    stats (dict): Various stats for each signal type.
  """
  sums = {}
  cnts = {}
  maxs = {}
  mins = {}
  avgs = {}
  stds = {}
  sdiff = {}
  stats = {}

  segment_bunches = [train_segments, valid_segments, test_segments]

  for bunch in segment_bunches:
    for segment in bunch:
      for k, v in segment.items():
        if k not in IGNORE:
          sums[k] = sums.get(k, 0) + np.sum(v)
          cnts[k] = cnts.get(k, 0) + v.size
          maxs[k] = np.max([maxs[k], np.max(v)]) if k in maxs else np.max(v)
          mins[k] = np.min([mins[k], np.min(v)]) if k in mins else np.min(v)

  for k in sums:
    avgs[k] = sums[k] / cnts[k]
  
  for bunch in segment_bunches:
    for segment in bunch:
      for k, v in segment.items():
        if k not in IGNORE:
          sdiff[k] = sdiff.get(k, 0) + ((v.flatten() - avgs[k]) ** 2).sum()

  for k in sums:
    stds[k] = np.sqrt(sdiff[k] / (cnts[k] - 1))

  for k in sums:
    stats[f'{k}_min'] = mins[k]
    stats[f'{k}_min'] = mins[k]
    stats[f'{k}_max'] = maxs[k]
    stats[f'{k}_std'] = stds[k]
    stats[f'{k}_avg'] = avgs[k]

  return stats


def dump_segments(path, segments):
  """
  Pickle segments to designed file path.

  Args:
    path (str): File path.
    segments (list[dict]): Segments.
  """
  with open(path, 'wb') as f:
    pickle.dump(segments, f)


def save_hemo_dataset(dataset_name):
  """
  Save list of train, validation, and test segments as pickle files.

  Args:
    dataset_name (str): Dataset name.
  """

  # Load RHC dataframe.
  rhc_path = os.path.join(DATABASE_PATH, 'meta_information', 'RHC_values.csv')
  rhc_df = pd.read_csv(rhc_path)

  # Get names of all records that may be included in a test set.
  all_test_records = get_test_record_names(DATABASE_PATH)

  # Randomly split test records into groups of a given length.
  all_groups = get_groups(all_test_records, NUM_TESTS)

  # Use first couple of batches for test set.
  test_record_groups = all_groups[:NUM_FOLDS]

  # Create different hold out sets.
  for i, test_records in enumerate(test_record_groups):

    # Identify training records by subtracting test records from all records.
    train_records = get_train_record_names(test_records, DATABASE_PATH)

    # Take a certain number of train records for the valid set.
    valid_records = train_records[:NUM_VALID]
    train_records = train_records[NUM_VALID:]

    # Define train, valid, and test directory paths.
    train_path = os.path.join('datasets', dataset_name, f'train_segments_{i+1}.pkl')
    valid_path = os.path.join('datasets', dataset_name, f'valid_segments_{i+1}.pkl')
    test_path = os.path.join('datasets', dataset_name, f'test_segments_{i+1}.pkl')

    # Define stats path.
    stats_path = os.path.join('datasets', dataset_name, f'global_stats_{i+1}.json')

    # Remove segment paths if already exists.
    if os.path.exists(train_path): os.remove(train_path)
    if os.path.exists(valid_path): os.remove(valid_path)
    if os.path.exists(test_path): os.remove(test_path)

    # Remove stats file if already exists.
    if os.path.exists(stats_path): os.remove(stats_path)

    # Get valid segments.
    valid_segments = get_dataset_segments(valid_records, rhc_df)
    dump_segments(valid_path, valid_segments)

    # Get test segments.
    test_segments = get_dataset_segments(test_records, rhc_df)
    dump_segments(test_path, test_segments)

    # Get train segments.
    train_segments = get_dataset_segments(train_records, rhc_df)
    dump_segments(train_path, train_segments)

    # Get global stats.
    global_stats = get_global_stats(train_segments, valid_segments, test_segments)
    with open(stats_path, 'w') as f:
      json_str = json.dumps(global_stats, indent=4)
      f.write(json_str)

    # Save segment info.
    log_path = os.path.join('datasets', dataset_name, f'segment_info_{i+1}.json')
    with open(log_path, 'w') as f:
      json_str = json.dumps({
        'train_records': train_records,
        'test_records': test_records,
        'valid_records': valid_records,
        'train_segments': len(train_segments),
        'valid_segments': len(valid_segments),
        'test_segments': len(test_segments),
      }, indent=4)
      f.write(json_str)


if __name__ == '__main__':
  dataset_name = sys.argv[1]
  save_hemo_dataset(dataset_name)
