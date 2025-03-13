import numpy as np
import os
import pandas as pd
import sys
from wave_test import get_pcc, get_rmse

def get_float_array(s):
  """
  Args:
    s (str): String representation of a list of floats.

  Returns:
    l (list[float]): The list object.
  """
  l = np.array([float(i) for i in s.strip('[').strip(']').split(', ')])
  return l



def get_epoch_stats(model_name, dataset_type):
  """
  Args:
    model_name (str): Model name.
    dataset_type (str): Dataset type (ie, valid or test).

  Returns:
    corrs (list[dict]): Epoch similarity stats.
  """
  corrs = []
  comparison_dir_path = os.path.join('models', model_name, 'comparisons', dataset_type)
  comparison_names = sorted(os.listdir(comparison_dir_path))
  
  for i, comparison_name in enumerate(comparison_names):
    all_pred = None
    all_real = None
    df = pd.read_csv(os.path.join(comparison_dir_path, comparison_name))
    
    for _, row in df.iterrows():
      pred_rhc = get_float_array(row['pred_rhc'])
      real_rhc = get_float_array(row['real_rhc'])
      all_pred = pred_rhc if all_pred is None else np.concatenate((all_pred, pred_rhc))
      all_real = real_rhc if all_real is None else np.concatenate((all_real, real_rhc))
    
    pcc_r, pcc_ci95_lower, pcc_ci95_upper = get_pcc(all_real, all_pred)
    rmse, rmse_ci95_lower, rmse_ci95_upper = get_rmse(all_real, all_pred)

    checkpoint = f"{comparison_name.split('.')[0]}.chk"
    corrs.append({
      'checkpoint': checkpoint,
      'pcc_r': pcc_r,
      'pcc_ci95_lower': pcc_ci95_lower,
      'pcc_ci95_upper': pcc_ci95_upper,
      'rmse': rmse,
      'rmse_ci95_lower': rmse_ci95_lower,
      'rmse_ci95_upper': rmse_ci95_upper
    })
    print(f'wave_epoch.py | {model_name} | {i+1}/{len(comparison_names)} | PCC: {pcc_r:.3f} [{pcc_ci95_lower:.3f}, {pcc_ci95_upper:.3f}] | RMSE: {rmse:.3f} [{rmse_ci95_lower:.3f}, {rmse_ci95_upper:.3f}]')
  return corrs


def run(model_name, dataset_type):
  """
  Calculate best epoch for given module and save results.

  Args:
    model_name (str): Model name.
    dataset_type (str): Dataset type (ie, valid or test).
  """
  print(f'Run wave_epoch.py for {model_name} {dataset_type} set')
  scores = get_epoch_stats(model_name, dataset_type)
  scores_df = pd.DataFrame.from_dict(scores)
  scores_df.to_csv(os.path.join('models', model_name, f'epoch_{dataset_type}_stats.csv'), index=False)
  best_score = scores_df.loc[scores_df['rmse'].idxmin()]
  with open(os.path.join('models', model_name, f'epoch_{dataset_type}_best.txt'), 'w') as f:
    f.write(best_score.to_string() + '\n')


if __name__ == '__main__':
  model_name = sys.argv[1]
  dataset_type = sys.argv[2]
  run(model_name, dataset_type)

