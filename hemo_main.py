from hemo_train import train

base_name = 'hemo_model_CO_2'
data_name = 'hemo_data_CO_2'
for fold in range(1, 6):
  model_name = f'{base_name}_{fold}'
  train(model_name, data_name, fold)
