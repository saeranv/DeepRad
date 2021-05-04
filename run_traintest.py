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
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

# sci py stuff
import numpy as np
import matplotlib.pyplot as plt
#from functools import partial


# set path
deeprad_dir = os.path.abspath(os.path.join(os.getcwd()))
if deeprad_dir not in sys.path:
  sys.path.insert(0, deeprad_dir)


# Import deeprad models
from deeprad.model import Autoencoder, CustomDataSet
from deeprad import utils
import deeprad.traintest_utils as traintest_utils
fd, pp = utils.fd, utils.pp

# Set seeds/device
np.random.seed(2)  # TODO: confirm right location?
device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
print("You are using device: %s" % device)

def main(hparam, checkpoint_dir=None, run_hparam=False, run_train=True, small_train=False):

  print('hparam:', hparam)

  if 'model_fname' not in hparam:
    # Extract epoch, batch_size for data calcs
    max_epochs, batch_size = hparam['max_epochs'], hparam['batch_size']


    # ---------------------------------------------------------------------------------
    # Define directories
    # ---------------------------------------------------------------------------------
    TARGET_FOLDER_PATH = os.path.join(deeprad_dir, "data", "traintest", "out_data")
    CHANNEL_FOLDER_PATH = os.path.join(deeprad_dir, "data", "traintest", "in_data")
    if small_train:
      TARGET_FOLDER_PATH = TARGET_FOLDER_PATH.replace('traintest', 'traintest_small')
      CHANNEL_FOLDER_PATH = CHANNEL_FOLDER_PATH.replace('traintest', 'traintest_small')

    # for opt
    checkpoint_dir = os.path.join(deeprad_dir, 'models', 'checkpoint')

    # Define model directory
    model_fname = 'model_'
    model_fname += '_'.join(['{}_{}'.format(n.replace('_', ''), np.round(v, 6))
      for n, v in hparam.items()])
    # model_fname = model_fname.replace('.', '_')  # deal with decimals
    model_dir = os.path.join(deeprad_dir, 'models', model_fname)
    if not os.path.isdir(model_dir):
      utils.make_dir_safely(model_dir)

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

  else:
    model_fname = hparam['model_fname']
    model_dir = os.path.join(deeprad_dir, 'models', model_fname)
    model_fpath, train_loss_fname, train_loss_arr_fpath, train_loss_img_fpath, \
      test_loss_fname, test_loss_arr_fpath, test_loss_img_fpath, \
        learning_loss_img_fpath = \
          traintest_utils.get_traintest_fpaths(deeprad_dir, model_dir, model_fname)


  #---------------------------------------------------------------------------------
  # Epoch iteration
  #---------------------------------------------------------------------------------
  print('\nTraining/Testing...')
  if not run_train:

    print('Loading train/test losses from {}.'.format(model_fname))
    train_loss_arr, test_loss_arr = np.load(train_loss_arr_fpath), np.load(test_loss_arr_fpath)

  else:

    print("Training/Testing {} data, over {} epochs".format(train_size, max_epochs))

    # Construct init model
    torch.manual_seed(42)
    model = Autoencoder(device, hparam['f1'], hparam['k1'])
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.Adam(
      model.parameters(), lr=hparam['learning_rate'], weight_decay=hparam['weight_decay'])

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
          test_loss += loss.item() / batch_size
          test_steps += 1

          if i < (test_iters - 1):
            del input_batch; del label_batch; del recon_batch; del loss; del i

      print('Test epoch:{}, Loss:{:.4f}'.format(epoch + 1, loss))
      outputs_test[epoch] = (label_batch, recon_batch)
      test_loss_arr[epoch] = test_loss / test_steps
      tune.report(test_loss=test_loss)

    print('Time took training/testing {}.'.format(utils.timer(start_time, time.time())))

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
  small_train = False
  run_train, run_hparam = True, True
  max_epochs, batch_size = 15, 5
  iter_dirs = False

  if len(sys.argv) > 1:
    argv = sys.argv[1:]
    if '--max_epochs' in argv:
      i = argv.index('--max_epochs')
      max_epochs = int(argv[i + 1])

    if '--run_train' in argv:
      i = argv.index('--run_train')
      run_train = bool(int(argv[i + 1]))

    if '--small_train' in argv:
      i = argv.index('--small_train')
      small_train = bool(int(argv[i + 1]))

    if '--batch_size' in argv:
      i = argv.index('--batch_size')
      batch_size = int(argv[i + 1])

    if '--run_hparam' in argv:
      i = argv.index('--run_hparam')
      run_hparam = bool(int(argv[i + 1]))

    if '--iter_dirs' in argv:
      i = argv.index('--iter_dirs')
      iter_dirs = bool(int(argv[i + 1]))

  print('User params (e#, b#, ts, rtrn, rtst, hpr): '
    '{}, {}, {}, {}, {}, {}'.format(
      max_epochs, batch_size, small_train, run_train, run_hparam, iter_dirs))

  # ---------------------------------------------------------------------------------
  # Hyperparameters
  # ---------------------------------------------------------------------------------

  if iter_dirs:
    model_dirs = os.listdir(os.path.join(deeprad_dir, 'models'))
    for model_dir in model_dirs:
      main({'model_fname': model_dir}, None, run_hparam=False, run_train=False, small_train=False)

  else:

    hparam = {
      'max_epochs': max_epochs,
      'batch_size': batch_size
    }

    if not run_hparam:
      hparam['learning_rate'] = 1e-3
      hparam['weight_decay'] = 1e-4
      main(hparam, None, run_hparam=run_hparam, run_train=run_train, small_train=small_train)

    else:

      # Best from initial trial
      # max_epoch = 15
      # batch_size = 5
      # learning_rate = 0.00184
      # weight_decay = 0.00020957

      hparam['f1'] = tune.choice([24, 32, 64])
      hparam['k1'] = tune.choice([3, 4, 6])
      hparam['learning_rate'] =  0.00184 # tune.loguniform(1e-4, 1e-1)
      hparam['weight_decay'] =  0.00020957 # tune.loguniform(1e-5, 1e-1)

      gpus_per_trial = 1
      cpu_num = 8
      num_samples = 2  # number of times to sample from parameter space

      result = tune.run(
        main,
        config=hparam,
        num_samples=num_samples,
        resources_per_trial={"cpu": cpu_num, "gpu": gpus_per_trial}
        )

      print(result.dataframe())
      # best_trial = result.get_best_trial("loss", "min", "last")
      # print("Best trial config: {}".format(best_trial.config))
      # print("Best trial final validation loss: {}".format(
      #     best_trial.last_result["loss"]))
      # print("Best trial final validation accuracy: {}".format(
      #     best_trial.last_result["accuracy"]))
