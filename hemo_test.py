import json
import numpy as np
import os
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from hemo_loader import get_loader
from hemo_record import NUM_STEPS, RHC_TYPE
from hemo_train import CardiovascularPredictor
from scipy.stats import pearsonr, t
from sklearn.metrics import mean_squared_error


def get_r2(x, y):
  """
  Args:
    x (ndarray): Test vector.
    y (ndarray): Real vector.

  Returns:
    r2 (float): Coefficient of determination.
    r2_ci95_lower (float): Lower limit of 95% CI.
    r2_ci95_upper (float): Upper limit of 95% CI.
  """
  result = pearsonr(x, y)
  pcc_r = result.statistic
  pcc_p = result.pvalue
  pcc_ci95 = result.confidence_interval(confidence_level=0.95)
  pcc_ci95_lower = pcc_ci95.low
  pcc_ci95_upper = pcc_ci95.high
  r2 = pcc_r[0] ** 2
  r2_ci95_lower = pcc_ci95_lower[0] ** 2
  r2_ci95_upper = pcc_ci95_upper[0] ** 2
  return r2, r2_ci95_lower, r2_ci95_upper
  

def get_rmse(x, y):
  """
  Args:
    x (ndarray): Test vector.
    y (ndarray): Real vector.

  Returns:
    rmse (float): Root mean squared error.
    pcc_ci95_lower (float): Lower limit of 95% CI.
    pcc_ci95_upper (float): Upper limit of 95% CI.
  """
  alpha = 0.05
  n = len(x)
  rmse = np.sqrt(mean_squared_error(x, y))
  se = np.sqrt(rmse / (2 * n))
  t_crit = t.ppf(1 - alpha/2, df=n-1)
  rmse_ci95_lower = rmse - t_crit * se
  rmse_ci95_upper = rmse + t_crit * se
  return rmse, rmse_ci95_lower, rmse_ci95_upper


def reverse_z_score_norm(v, gs):
  """
  Args:
    v (ndarray): Vector.
    gs (dict): Contains stats across all segments in a dataset.

  Returns:
    (ndarray): Vector with reversed Z-score normalization.
  """
  return v * gs[f'{RHC_TYPE}_std'] + gs[f'{RHC_TYPE}_avg']


def test(model_name, data_name, data_fold):
  """
  Show net results of a model over multiple folds.

  Args:
    model_name (str): Model base name.
    data_name (str): Dataset directory name.
    data_fold (str): Dataset folds to include, either a single number or 'all'.
  """
  
  # Determine which folds to include.
  data_folds = []
  if data_fold == 'all':
    test_fold = 1
    while True:
      test_path = os.path.join('datasets', data_name, 
                              f'global_stats_{test_fold}.json')
      if os.path.exists(test_path):
        data_folds.append(test_fold)
      else:
        break
      test_fold += 1
  else:
    data_folds = [int(data_fold)]
  
  # Initialize net real and predicted vectors.
  all_real = None
  all_pred = None
  
  # Iterate through each data fold, concatenating all real and predicted vectors.
  for data_fold in data_folds:
    
    model_best_path = os.path.join('models', 
                                  f'{model_name}_{data_fold}', 
                                   'model_best.chk')
    checkpoint = torch.load(model_best_path, weights_only=False)
    model = CardiovascularPredictor()
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f'Model = {model_name}_{data_fold}')
    print(f"Best Valid Loss = {checkpoint['min_valid_loss']}")

    # Load global stats for vector normalization.
    stats_path = os.path.join('datasets', data_name, 
                             f'global_stats_{data_fold}.json')
    
    with open(stats_path, 'r') as f:
      global_stats = json.load(f)
    
    # Load test segments.
    test_loader_path = os.path.join('datasets', data_name, 
                                   f'test_segments_{data_fold}.pkl')
    
    test_loader = get_loader(test_loader_path, global_stats, 128)
    
    # Iterate through all test segments.
    for i, (acc, ecg, bmi, label) in enumerate(test_loader, start=1):

      # Get raw predicted and actual RHC value.
      real_y = label.detach().numpy().squeeze(-1)
      pred_y = model(acc, ecg, bmi).detach().numpy()

      # Reverse z-score normalization.
      real_y = reverse_z_score_norm(real_y, global_stats)
      pred_y = reverse_z_score_norm(pred_y, global_stats)

      # Concatenate RHC values.
      all_real = real_y if all_real is None else np.concat((all_real, real_y))
      all_pred = pred_y if all_pred is None else np.concat((all_pred, pred_y))

      # Calculate root mean squared error and pearson correlation coefficient.
      rmse, rmse_ci95_lower, rmse_ci95_upper = get_rmse(all_real, all_pred)
      r2, r2_ci95_lower, r2_ci95_upper = get_r2(all_real, all_pred)

      print(f'Fold = {data_fold}, Batch = {i}/{len(test_loader)}, '
            f'RMSE = {rmse:.2f}, R2 = {r2:.2f}') 
  
  results = {
    'rmse': rmse,
    'rmse_ci95_lower': rmse_ci95_lower,
    'rmse_ci95_upper': rmse_ci95_upper,
    'r2': r2,
    'r2_ci95_lower': r2_ci95_lower,
    'r2_ci95_upper': r2_ci95_upper
  }

  for k, v in results.items():
    print(f'{k} = {v}')


if __name__ == '__main__':
  model_name = sys.argv[1]
  data_name = sys.argv[2]
  data_fold = sys.argv[3]
  test(model_name, data_name, data_fold)
