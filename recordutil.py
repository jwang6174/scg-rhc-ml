import json
import numpy as np
import os
import pickle
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


def get_dataset_segments(acc_channels, chamber, segment_size, flat_amp_threshold, 
                         flat_min_duration, straight_threshold, min_RHC, max_RHC,
                         db_path, sample_rate):
  """
  Args:
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
  for record_name in get_record_names(db_path):
    record_segments = get_record_segments(record_name, acc_channels, chamber, 
                                          segment_size, flat_amp_threshold, 
                                          flat_min_duration, straight_threshold, 
                                          min_RHC, max_RHC, db_path, sample_rate)
    dataset_segments.extend(record_segments)
  return dataset_segments


def save_dataset(dataset_name, acc_channels, chamber, segment_size, 
                 train_size, flat_amp_threshold, flat_min_duration, 
                 straight_threshold, min_RHC, max_RHC, db_path, sample_rate):
  """
  Save list of train, validation, and test segments as pickle files.

  Args:
    dataset_name (str): Dataset name.
    batch_size (int): Batch size.
    acc_channels (list[str]): ACC channels.
    RHC_col (str): RHC column of interest when performing value prediction.
    chamber (str): Heart chamber of interest when performing waveform prediction.
    segment_size (float): Segment duration (sec).
    train_size (float): Percentage of segments that should be used for training.
    flat_amp_threshold (float): Amplitude threshold to be considered flat.
    flat_min_duration (float): Minimum duration (seconds) to be considered flat.
    straight_threshold (float): R squared threshold to be considered straight line.
    min_RHC (float): Minimum allowable RHC value.
    max_RHC (float): Maximum allowable RHC value.
    db_path (str): SCG-RHC database path.
    sample_rate (int): Sample rate (Hz).
  """
  print(f'Run recordutil.py for {dataset_name}')
  
  dataset_segments = get_dataset_segments(acc_channels, chamber, segment_size,
                                          flat_amp_threshold, flat_min_duration, 
                                          straight_threshold, min_RHC, max_RHC,
                                          db_path, sample_rate)
  
  train_segments, non_train_segments = train_test_split(dataset_segments,
                                                        shuffle=True,
                                                        train_size=train_size)
  
  valid_segments, test_segments = train_test_split(non_train_segments, 
                                                   shuffle=True,
                                                   train_size=0.5)
    
  train_path = os.path.join('datasets', dataset_name, 'train_segments.pkl')
  with open(train_path, 'wb') as f:
    pickle.dump(train_segments, f)

  valid_path = os.path.join('datasets', dataset_name, 'valid_segments.pkl')
  with open(valid_path, 'wb') as f:
    pickle.dump(valid_segments, f)

  test_path = os.path.join('datasets', dataset_name, 'test_segments.pkl')
  with open(test_path, 'wb') as f:
    pickle.dump(test_segments, f)

  log_path = os.path.join('datasets', dataset_name, 'segment_info.txt')
  with open(log_path, 'w') as f:
    json_str = json.dumps({
      'train_segments': len(train_segments),
      'valid_segments': len(valid_segments),
      'test_segments': len(test_segments),
    }, indent=4)
    f.write(json_str)


if __name__ == '__main__':
  dataset_name = sys.argv[1]
  dataset_params_path = os.path.join('datasets', dataset_name, 'params.json')
  with open(dataset_params_path, 'r') as f:
    params = json.load(f)
    save_dataset(
      dataset_name,
      params['acc_channels'],
      params['chamber'],
      params['segment_size'],
      params['train_size'],
      params['flat_amp_threshold'],
      params['flat_min_duration'],
      params['straight_threshold'],
      params['min_RHC'],
      params['max_RHC'],
      DATABASE_PATH,
      SAMPLE_RATE
    )
