import json
import numpy as np
import os
import pandas as pd
import pickle
import sys
import torch
from loaderutil import get_loader
from recordutil import SAMPLE_RATE
from scipy.stats import pearsonr, t
from sklearn.metrics import mean_squared_error
from time import time
from timeutil import timelog
from torch.utils.data import DataLoader
from wave_train import Generator


def reverse_minmax(signal, orig_min, orig_max):
  """
  Args:
    signal (array_like): Signal already min-max normalized.
    orig_min (float_like): Original min value.
    orig_max (float_like): Original max value.

  Returns:
    signal (array_like): Original signal before min-max normalization.
  """
  return (signal * (float(orig_max) - float(orig_min))) + float(orig_min)


def get_pcc(x, y):
  """
  Args:
    x (array_like): Input array.
    y (array_like): Input array.

  Returns:
    pcc_r (float): Pearson correlation coefficient.
    pcc_ci95_lower (float): Lower bound of 95% CI.
    pcc_ci95_upper (float): Upper bound of 95% CI.
  """
  result = pearsonr(x, y)
  pcc_r = result.statistic
  pcc_p = result.pvalue
  pcc_ci95 = result.confidence_interval(confidence_level=0.95)
  pcc_ci95_lower = pcc_ci95.low
  pcc_ci95_upper = pcc_ci95.high
  return pcc_r, pcc_ci95_lower, pcc_ci95_upper


def get_rmse(x, y):
  """
  Args:
    x (array_like): Input array.
    y (array_like): Input array.

  Returns:
    rmse (float): Root mean squared error.
    rmse_ci95_lower (float): Lower bound of 95% CI.
    rmse_ci95_upper (float): Upper bound of 95% CI.
  """
  alpha = 0.05
  n = len(x)
  rmse = np.sqrt(mean_squared_error(x, y))
  se = np.sqrt(rmse / (2 * n))
  t_crit = t.ppf(1 - alpha/2, df=n-1)
  rmse_ci95_lower = rmse - t_crit * se
  rmse_ci95_upper = rmse + t_crit * se
  return rmse, rmse_ci95_lower, rmse_ci95_upper


def get_wave_comparisons(generator, loader):
  """
  Args:
    generator (Module): Generator.
    loader (DataLoader): Data loader.

  Returns:
    comparisons (list[dict]): Segment info and similarity scores.
  """
  comparisons = []
  for i, segment in enumerate(loader):

    x = segment['rhc'].detach().numpy()[0, 0, :]
    x = reverse_minmax(x, segment['rhc_min'], segment['rhc_max'])

    y = generator(segment['acc']).detach().numpy()[0, 0, :]
    y = reverse_minmax(y, segment['rhc_min'], segment['rhc_max'])

    pcc_r, pcc_ci95_lower, pcc_ci95_upper = get_pcc(x, y)
    rmse, rmse_ci95_lower, rmse_ci95_upper = get_rmse(x, y)

    comparison = {
      'record_name': segment['record_name'],
      'start_idx': segment['start_idx'],
      'end_idx': segment['end_idx'],
      'real_rhc': str(x.tolist()),
      'pred_rhc': str(y.tolist()),
      'pcc_r': pcc_r,
      'pcc_ci95_lower': pcc_ci95_lower,
      'pcc_ci95_upper': pcc_ci95_upper,
      'rmse': rmse,
      'rmse_ci95_lower': rmse_ci95_lower,
      'rmse_ci95_upper': rmse_ci95_upper,
    }

    comparisons.append(comparison)

  return comparisons


def get_processed_checkpoint_names(comparison_dir_path):
  """
  Args:
    comparison_dir_path (str): Comparison directory path.

  Returns:
    (frozenset): Processed checkpoint filenames.
  """
  return frozenset(f"{f.split('.')[0]}.chk" for f in os.listdir(
                                                      comparison_dir_path))


def run(model_name, dataset_type, checkpoint_mode):
  """
  Generate similarity stats between real and predicted RHC waveforms. Saves a
  Pandas dataframe for each checkpoint.

  Args:
    model_name (str): Model name.
    
    dataset_type (str): Dataset type (ie, train, valid, or test).
    
    checkpoint_mode (str): Set which checkpoints should be analyzd ('all' to
    run over all checkpoints, 'best' to run over the best performing checkpoint
    on the valid set, or the name of a specific checkpoint).
  """
  start_time = time()

  # Get model params.
  model_path = os.path.join('models', model_name, 'params.json')
  with open(model_path, 'r') as f:
    model_params = json.load(f)

  # Get dataset name and fold.
  dataset_name = model_params['dataset_name']
  dataset_fold = model_params['dataset_fold']

  # Get dataset parms.
  dataset_path = os.path.join('datasets', dataset_name, 'params.json')
  with open(dataset_path, 'r') as f:
    dataset_params = json.load(f)

  # Get global stats.
  global_stats_path = os.path.join('datasets', dataset_name, 
                                  f'global_stats_{dataset_fold}.json')
  with open(global_stats_path, 'r') as f:
    global_stats = json.load(f)

  # Get segments and loader.
  segments_name = f'{dataset_type}_segments_{dataset_fold}.pkl'
  segments_path = os.path.join('datasets', dataset_name, segments_name)
  with open(segments_path, 'rb') as f:
    segments = pickle.load(f)
    loader = get_loader(segments, 
                        dataset_params['segment_size'] * SAMPLE_RATE, 
                        1, 
                        global_stats, 
                        model_params['norm_type'])


  # Define checkpoints and comparisons directory paths.
  checkpoint_dir_path = os.path.join('models', model_name, 'checkpoints')
  comparison_dir_path = os.path.join('models', model_name, 'comparisons', dataset_type)
  
  # Create comparisons directories if not exists.
  if not os.path.exists(comparison_dir_path):
    os.makedirs(comparison_dir_path)

  # Get all checkpoint paths if specified.
  if checkpoint_mode == 'all':
    checkpoint_names = set((os.listdir(checkpoint_dir_path)))

  # Get best checkpoint determined on valid set.
  elif checkpoint_mode == 'best':
    with open(os.path.join('models', model_name, 'epoch_valid_best.txt'), 'r') as f:
      checkpoint_name = f.read().splitlines()[0].split()[1]
      checkpoint_names = [checkpoint_name]

  # Otherwise use specific checkpoint.
  else:
    checkpoint_names = [checkpoint_mode]

  # Get processed checkpoint names.
  processed_checkpoint_names = set(get_processed_checkpoint_names(comparison_dir_path))

  # Iterate through each checkpoint, calculate PCC and RMSE, and output
  # checkpoint with lowest RMSE.
  for i, checkpoint_name in enumerate(checkpoint_names):

    timelog(f'wave_test.py | {model_name} | {dataset_type} | '
            f'{i+1}/{len(checkpoint_names)}', start_time)

    if checkpoint_name in processed_checkpoint_names:
      continue

    checkpoint_path = os.path.join(checkpoint_dir_path, checkpoint_name)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    num_in_channels = len(dataset_params['acc_channels'])
    generator = Generator(num_in_channels)
    generator.load_state_dict(checkpoint['g_state_dict'])
    generator.eval()

    comparisons = get_wave_comparisons(generator, loader)
    comparisons.sort(key=lambda x: x['pcc_r'], reverse=True)

    checkpoint_str = checkpoint_name.split('.')[0]
    comparison_path = os.path.join(comparison_dir_path, 
                                f'{checkpoint_str}.csv')

    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(comparison_path, index=False)



if __name__ =='__main__':
  model_name = sys.argv[1]
  dataset_type = sys.argv[2]
  checkpoint_mode = sys.argv[3]
  run(model_name, dataset_type, checkpoint_mode)


