import json
import matplotlib.pyplot as plt
import os
import pickle
import sys
import torch
from loaderutil import get_loader, SCGDataset
from recordutil import SAMPLE_RATE
from wave_train import Generator

def save_rand_RHC_plots(model_name, checkpoint_name, dataset_type, num_plots, 
                        sample_rate):
  """
  Save a number of random predicted and real RHC plots.

  Args:
    model_name (str): Model name.
    checkpoint_name (str): Model checkpoint name.
    dataset_type (str): Dataset type (ie, train, valid, or test).
    num_plots (int): Number of plots to generate.
    sample_rate (int): Sample rate (Hz).
  """

  # Get model params.
  model_path = os.path.join('models', model_name, 'params.json')
  with open(model_path, 'r') as f:
    model_params = json.load(f)

  # Get dataset name andn fold.
  dataset_name = model_params['dataset_name']
  dataset_fold = model_params['dataset_fold']

  # Get dataset params.
  dataset_path = os.path.join('datasets', dataset_name, 'params.json')
  with open(dataset_path, 'r') as f:
    dataset_params = json.load(f)

  # Get global stats.
  stats_name = f'global_stats_{dataset_fold}.json'
  stats_path = os.path.join('datasets', dataset_name, stats_name)
  with open(stats_path, 'r') as f:
    global_stats = json.load(f)

  # Get segments and loader.
  segments_name = f'valid_segments_{dataset_fold}.pkl'
  segments_path = os.path.join('datasets', dataset_name, segments_name)
  with open(segments_path, 'rb') as f:
    segments = pickle.load(f)
    loader = get_loader(segments, 
                        dataset_params['segment_size'] * sample_rate, 
                        1,
                        global_stats,
                        model_params['norm_type'])

  # Create plots directory if not exists.
  plots_dir_path = os.path.join('models', model_name, 'plots')
  if not os.path.exists(plots_dir_path):
    os.makedirs(plots_dir_path)

  # Get model checkpoint.
  checkpoint_path = os.path.join('models', model_name, 'checkpoints', 
                                 f'{checkpoint_name}.chk')
  checkpoint = torch.load(checkpoint_path, weights_only=False)

  # Get generator.
  num_in_channels = len(dataset_params['acc_channels'])
  generator = Generator(num_in_channels)
  generator.load_state_dict(checkpoint['g_state_dict'])
  generator.eval()

  # Create plots.
  for i, segment in enumerate(loader):
    if i == num_plots:
      break
    acc = segment['acc']
    rhc = segment['rhc']
    real_rhc = rhc.detach().numpy()[0, 0, :]
    pred_rhc = generator(acc).detach().numpy()[0, 0, :]
    plt.plot(real_rhc, label='Real RHC')
    plt.plot(pred_rhc, label='Pred RHC')
    plt.xlabel('Sample')
    plt.ylabel('Norm')
    plt.legend()
    plot_name = f'rand_RHC_plot_{dataset_type}_{checkpoint_name}_{i}.png'
    plt.savefig(os.path.join(plots_dir_path, plot_name))
    plt.close()


if __name__ == '__main__':
  model_name = sys.argv[1]
  checkpoint_name = sys.argv[2]
  dataset_type = sys.argv[3]
  num_plots = int(sys.argv[4])
  save_rand_RHC_plots(model_name, checkpoint_name, dataset_type, num_plots, 
                      SAMPLE_RATE)
