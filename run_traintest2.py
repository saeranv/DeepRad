import sys
import os
import glob
import time

# torch
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

gc.collect()
torch.cuda.empty_cache()

from ray import tune
#from ray.tune import CLIReporter
#from ray.tune.schedulers import ASHAScheduler
#from ray.tune.suggest.hyperopt import HyperOptSearch

# sci py stuff
import numpy as np
import matplotlib.pyplot as plt

# set path
deeprad_dir = os.path.abspath(os.path.join(os.getcwd()))
if deeprad_dir not in sys.path:
  sys.path.insert(0, deeprad_dir)

# Import deeprad models
from deeprad.model import Autoencoder, CustomDataSet
from deeprad import utils
import deeprad.traintest_utils as traintest_utils
fd, pp = utils.fd, utils.pp

def main(hparam, checkpoint_dir=None, run_hparam=False, small_train=False):

  # Set seeds/device
  np.random.seed(2)  # TODO: confirm right location?
  device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
  print("You are using device: %s" % device)
  print('hparam:', hparam)

  # Extract epoch, batch_size for data calcs
  max_epochs, batch_size = hparam['max_epochs'], hparam['batch_size']
  # ---------------------------------------------------------------------------------
  # Define directories
  # ---------------------------------------------------------------------------------
  TARGET_FOLDER_PATH = os.path.join(deeprad_dir, "data", "traintest2", "out_data2")
  CHANNEL_FOLDER_PATH = os.path.join(deeprad_dir, "data", "traintest2", "in_data2")
  if small_train:
    TARGET_FOLDER_PATH = TARGET_FOLDER_PATH.replace('traintest2', 'traintest_small2')
    CHANNEL_FOLDER_PATH = CHANNEL_FOLDER_PATH.replace('traintest2', 'traintest_small2')

  # for opt
  checkpoint_dir = os.path.join(deeprad_dir, 'models', 'checkpoint')

  # Define model directory
  now = time.strftime("%Y%m%d-%H%M%S")
  nullify = 'delete_me' if small_train else 'model'
  model_fname = '{}2_{}'.format(nullify, now)

  # model_pars += '_'.join(['{}_{}'.format(n.replace('_', ''), np.round(v, 6))
  #   for n, v in hparam.items()])
  # model_fname = model_fname.replace('.', '_')  # deal with decimals
  model_dir = os.path.join(deeprad_dir, 'models', model_fname)
  if not os.path.isdir(model_dir):
    utils.make_dir_safely(model_dir)

  with open(os.path.join(model_dir, 'hparams.txt'), 'w') as hparam_file:
    for k, v in hparam.items():
      hparam_file.write('{}: {}\n'.format(k, v))

  # Define model, loss, image fpaths
  model_fpath, train_loss_fname, train_loss_arr_fpath, train_loss_img_fpath, \
    test_loss_fname, test_loss_arr_fpath, test_loss_img_fpath, \
      learning_loss_img_fpath = \
        traintest_utils.get_traintest_fpaths(deeprad_dir, model_dir, model_fname)

  # ---------------------------------------------------------------------------------
  # Set data
  # ---------------------------------------------------------------------------------

  # Load training/test set
  dataset = CustomDataSet(TARGET_FOLDER_PATH, CHANNEL_FOLDER_PATH,
                          transform=transforms.ToTensor(), device=device)
  train_size = int(len(dataset) * .8)
  test_size = len(dataset) - train_size
  train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
  train_iters, test_iters = int(train_size // batch_size), int(test_size // batch_size)

  print('data-{}; train={} + test={}; batch={}'.format(len(dataset), train_size, test_size, batch_size))

  if (test_size / batch_size) < 1.0:
    raise Exception('batch size {} too large for test size: {}'.format(test_size, batch_size))

  # for use in training and testing
  train_criterion = nn.MSELoss() # mean square error loss
  test_criterion = nn.MSELoss()

  # ---------------------------------------------------------------------------------
  # Load Pretrainted Model
  # ---------------------------------------------------------------------------------
  # Construct init model
  torch.manual_seed(42)

  # model = Autoencoder(device, hparam['f1'], hparam['k1'])

  pretrained_model_fpath = os.path.join(
    deeprad_dir, 'models', 'model_best_cnn2', 'model_best_cnn2.pt')
  print(pretrained_model_fpath)
  assert os.path.isfile(pretrained_model_fpath)
  model = torch.load(pretrained_model_fpath)
  model.eval()

  if torch.cuda.device_count() > 1:
      model = nn.DataParallel(model)
  model = model.to(device)

  # Freeze encoder
  # model.encoder.requires_grad = False

  # optimizer
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
    lr=hparam['learning_rate'], weight_decay=hparam['weight_decay'])

  #---------------------------------------------------------------------------------
  # Epoch iteration
  #---------------------------------------------------------------------------------
  print("Training/Testing {} data, over {} epochs".format(train_size, max_epochs))

  # Training init
  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
  outputs_train = [0] * max_epochs
  train_loss_arr = np.zeros((max_epochs, 1))

  # Testing init
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)
  outputs_test = [0] * max_epochs
  test_loss_arr = np.zeros((max_epochs, 1))


  # Start epoch iterations
  start_time = time.time()
  for epoch in range(max_epochs):
    #---------------------------
    # training
    #---------------------------
    train_loss, train_steps = 0, 0
    for i, (input_batch, label_batch) in enumerate(train_loader):
        input_batch, label_batch = input_batch.to(device), label_batch.to(device)

        # forward pass
        recon_batch = model.forward(input_batch)
        loss = train_criterion(recon_batch, label_batch)
        train_loss += loss.item() / batch_size
        train_steps += 1

        # calc gradient with autograd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i < (train_iters - 1):
          del input_batch; del label_batch; del recon_batch; del loss; del i

    print('Train epoch:{}, Loss:{:.4f}'.format(epoch + 1, loss))
    outputs_train[epoch] = (label_batch, recon_batch)
    train_loss_arr[epoch] = train_loss / train_steps

    # ---------------------------
    # Testing
    # ---------------------------
    test_loss, test_steps = 0, 0
    for i, (input_batch, label_batch) in enumerate(test_loader):
        input_batch, label_batch = input_batch.to(device), label_batch.to(device)
        recon_batch = model(input_batch)

        # calc loss
        loss = test_criterion(recon_batch, label_batch)
        test_loss += loss.cpu().item() / batch_size
        test_steps += 1

        if i < (test_iters - 1):
          del input_batch; del label_batch; del recon_batch; del loss; del i

    print('Test epoch:{}, Loss:{:.4f}'.format(epoch + 1, loss))
    inter_time = utils.timer(start_time, time.time())
    print('time: {}.'.format(inter_time))

    outputs_test[epoch] = (label_batch, recon_batch)
    test_loss_arr[epoch] = test_loss / test_steps

    if run_hparam:
      tune.track.log(test_loss=test_loss_arr[-1])
      torch.save(model.state_dict(), model_fpath)

  end_time = utils.timer(start_time, time.time())
  print('Time took training/testing {}.'.format(end_time))
  with open(os.path.join(model_dir, 'hparams.txt'), 'a') as hparam_file:
    hparam_file.write('last test_loss: {}\n'.format(test_loss_arr[-1]))
    hparam_file.write('test_loss: {}\n'.format(test_loss_arr))
    hparam_file.write('training time: {}\n'.format(end_time))

  # ---------------------------
  # Save data
  # ---------------------------
  # Save training model, image, losses
  torch.save(model, model_fpath)
  np.save(train_loss_arr_fpath, train_loss_arr)
  traintest_utils.viz_loss(
    outputs_train, img_fpath=train_loss_img_fpath, img_title='Train Loss', show_plot=False)
  print('Saved training model, loss image and data: {}'.format(model_fname))

  # Save test image, losses
  np.save(test_loss_arr_fpath, test_loss_arr)
  traintest_utils.viz_loss(
    outputs_test, img_fpath=test_loss_img_fpath, img_title='Test Loss', show_plot=False)
  print('Saved testing model, loss image and data: {}'.format(model_fname))

  # ---------------------------------------------------------------------------------
  # Plotting Learning
  # ---------------------------------------------------------------------------------
  traintest_utils.viz_learning_curve(
    train_loss_arr, test_loss_arr, model_fname, learning_loss_img_fpath)

  print('\nFinished\n')


if __name__ == "__main__":
  arg_dict = {
    'max_epochs': 20,
    'small_train': False,
    'batch_size': 5,
    'run_hparam': False,
    }

  if len(sys.argv) > 1:
    argv = sys.argv[1:]

    for arg in argv:
      if '--' not in arg: continue
      assert arg.split('--')[-1] in arg_dict, \
        '{} not an arg: {}'.format(arg, arg_dict.keys())

    if '--max_epochs' in argv:
      i = argv.index('--max_epochs')
      arg_dict['max_epochs'] = int(argv[i + 1])

    if '--small_train' in argv:
      i = argv.index('--small_train')
      arg_dict['small_train'] = bool(int(argv[i + 1]))

    if '--batch_size' in argv:
      i = argv.index('--batch_size')
      arg_dict['batch_size'] = int(argv[i + 1])

    if '--run_hparam' in argv:
      i = argv.index('--run_hparam')
      arg_dict['run_hparam'] = bool(int(argv[i + 1]))


  print('\nPress Enter to confirm user arg:')
  [print('{}: {}'.format(k, v)) for k, v in arg_dict.items()]
  input('...')

  # ---------------------------------------------------------------------------------
  # Hyperparameters
  # ---------------------------------------------------------------------------------

  max_epochs = arg_dict['max_epochs']
  small_train = arg_dict['small_train']
  batch_size = arg_dict['batch_size']
  run_hparam = arg_dict['run_hparam']

  hparam = {
    'max_epochs': max_epochs,
    'batch_size': batch_size
  }

  if not run_hparam:
    # "learning_rate\": 0.000203228544324115,\n
    # \"weight_decay\": 1.2697111322756407e-05\
    hparam['learning_rate'] = 0.000203228544324115 #0.00184
    hparam['weight_decay'] = 1.2697111322756407e-05 #0.00020957

    hparam['f1'] = 24
    hparam['k1'] = 3
    main(hparam, None, run_hparam=run_hparam, small_train=small_train)

  else:

    # Best from initial trial
    # max_epoch = 15
    # batch_size = 5
    # learning_rate = 0.00184
    # weight_decay = 0.00020957

    hparam['f1'] = 24 #tune.grid_search([24, 48, 64, 128])
    hparam['k1'] = 3 #3 #tune.grid_search([3, 4, 6])
    hparam['learning_rate'] = tune.uniform(1e-5, 1e-2)
    hparam['weight_decay'] = tune.uniform(1e-5, 1e-1)

    gpus_per_trial = 1
    cpu_num = 2
    num_samples = 4  # number of times to sample from parameter space

    result = tune.run(
      main,
      stop={"training_iteration": 5},
      config=hparam,
      num_samples=num_samples,
      resources_per_trial={"cpu": cpu_num, "gpu": gpus_per_trial}
      )

    # tensorboard --logdir ~/ray_results
    print(result.dataframe())

    result.dataframe().to_csv(os.path.join(deeprad_dir, 'models', 'df', 'result_df.csv'))