import sys
import os
import numpy as np

from ray import tune
#from ray.tune import CLIReporter
#from ray.tune.schedulers import ASHAScheduler
#from ray.tune.suggest.hyperopt import HyperOptSearch

# set path
deeprad_dir = os.path.abspath(os.path.join(os.getcwd()))
if deeprad_dir not in sys.path:
  sys.path.insert(0, deeprad_dir)

# Import deeprad models
from deeprad import traintest_utils
from deeprad.model import Autoencoder
from deeprad import utils
pp, fd = utils.pp, utils.fd

np.random.seed(2)  # TODO: confirm right location?

import torch

def parse_argv(arg_str, argv, arg_dict, is_bool=False):

  search_arg_str = '--' + arg_str
  if search_arg_str in argv:
    i = argv.index(search_arg_str)
    return bool(int(argv[i + 1])) if is_bool else int(argv[i + 1])
  else:
    return arg_dict[arg_str]

def update_arg_dict(arg_dict):
  """Parse user args and add to dictionary."""
  if len(sys.argv) > 1:
    argv = sys.argv[1:]

    for arg in argv:
      if '--' not in arg: continue
      assert arg.split('--')[-1] in arg_dict, \
        '{} not an arg: {}'.format(arg, arg_dict.keys())

    arg_dict['max_epochs'] = parse_argv('max_epochs', argv, arg_dict, is_bool=False)
    arg_dict['batch_size'] = parse_argv('batch_size', argv, arg_dict, is_bool=False)
    arg_dict['data_limit'] = parse_argv('data_limit', argv, arg_dict, is_bool=False)
    arg_dict['run_hparam'] = parse_argv('run_hparam', argv, arg_dict, is_bool=True)
    arg_dict['run_gpu'] = parse_argv('run_gpu', argv, arg_dict, is_bool=True)

  return arg_dict

if __name__ == "__main__":
  torch.cuda.empty_cache()
  arg_dict = {
    'max_epochs': 25,
    'batch_size': 5,
    'data_limit': None,
    'run_hparam': False,
    'run_gpu': True}

  try:
    arg_dict = update_arg_dict(arg_dict)
    print('\nPress Enter to confirm user arg:')
    [print('\t{}: {}'.format(k, v)) for k, v in arg_dict.items()]
    input('...')
  except Exception as e:
    print('Skip arg dict. {}'.format(e))


  # ---------------------------------------------------------------------------------
  # Hyperparameters
  # ---------------------------------------------------------------------------------

  out_dir = os.path.join(deeprad_dir, "data", "traintest3", "out_data")
  in_dir = os.path.join(deeprad_dir, "data", "traintest3", "in_data")

  max_epochs = arg_dict['max_epochs']
  batch_size = arg_dict['batch_size']
  data_limit = arg_dict['data_limit']
  run_hparam = arg_dict['run_hparam']
  run_gpu = arg_dict['run_gpu']

  hparam = {
    'max_epochs': max_epochs,
    'batch_size': batch_size}

  # Define model directory
  now = utils.time_str()
  nullify = 'delete_me' if data_limit else 'model'
  model_fname = '{}_source_{}'.format(nullify, now)

  if not run_hparam:
    hparam['learning_rate'] = 0.002 #0.000203228544324115 #0.00184
    hparam['weight_decay'] = 1.2697111322756407e-05 #0.00020957
    hparam['f1'] = 16
    hparam['k1'] = 3
    model, outputs_test, train_loss_arr, test_loss_arr, out_fpaths = traintest_utils.main(
         hparam, model_fname, Autoencoder, checkpoint_dir=None, in_dir=in_dir,
         out_dir=out_dir, run_hparam=run_hparam, data_limit=data_limit, run_gpu=run_gpu,
         pretrained_model_fpath=None) #'model_best_cnn2')

    # ---------------------------
    # Save data
    # ---------------------------
    model_fpath, test_loss_img_fpath, learning_loss_img_fpath = out_fpaths
    torch.save(model, model_fpath)

    # Save test image, losses
    # np.save(test_loss_arr_fpath, test_loss_arr)
    traintest_utils.viz_loss(
      outputs_test, img_fpath=test_loss_img_fpath, img_title='Test Loss', show_plot=False)
    print('Saved testing model, loss image and data: {}'.format(model_fname))

    # ---------------------------
    # Save data
    # ---------------------------
    traintest_utils.viz_learning_curve(
      train_loss_arr, test_loss_arr, model_fname, learning_loss_img_fpath)

  else:

    hparam['f1'] = tune.grid_search([24, 48, 64, 128])
    hparam['k1'] = 3 #tune.grid_search([3, 4, 6])
    hparam['learning_rate'] = 0.000203228544324115 # tune.uniform(1e-5, 1e-2)
    hparam['weight_decay'] = 1.2697111322756407e-05 # tune.uniform(1e-5, 1e-1)

    gpus_per_trial = 1
    cpu_num = 2
    num_samples = 4  # number of times to sample from parameter space

    result = tune.run(
      main,
      stop={"training_iteration": 5},
      config=hparam,
      num_samples=num_samples,
      resources_per_trial={"cpu": cpu_num, "gpu": gpus_per_trial})

    # tensorboard --logdir ~/ray_results
    print(result.dataframe())

    result.dataframe().to_csv(
      os.path.join(deeprad_dir, 'models', '{}_result_df.csv'.format(utils.time_str())))

