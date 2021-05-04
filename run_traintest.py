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

# raytune
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler


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

# Set seeds/device
np.random.seed(2)  # TODO: confirm right location?
device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
print("You are using device: %s" % device)

# Data folders
TARGET_FOLDER_PATH = os.path.join(deeprad_dir, "data", "traintest", "out_data")
CHANNEL_FOLDER_PATH = os.path.join(deeprad_dir, "data", "traintest", "in_data")
assert os.path.isdir(TARGET_FOLDER_PATH) and os.path.isdir(CHANNEL_FOLDER_PATH)


def main(max_epochs=30, batch_size=20, run_train=True, run_test=True, small_train=False):

  # Data folders
  TARGET_FOLDER_PATH = os.path.join(deeprad_dir, "data", "traintest", "out_data")
  CHANNEL_FOLDER_PATH = os.path.join(deeprad_dir, "data", "traintest", "in_data")
  assert os.path.isdir(TARGET_FOLDER_PATH) and os.path.isdir(CHANNEL_FOLDER_PATH)


  # Hyperparameters
  max_epochs = max_epochs
  learning_rate = 1e-3
  weight_decay = 1e-5
  batch_size = batch_size # images trained at once in layer
  # TODO: SGD/Adam/error

  if small_train:
    TARGET_FOLDER_PATH = TARGET_FOLDER_PATH.replace('traintest', 'traintest_small')
    CHANNEL_FOLDER_PATH = CHANNEL_FOLDER_PATH.replace('traintest', 'traintest_small')

  # Define model directory
  model_fname = 'model_{}'.format('epoch_{}_lr_{}'.format(max_epochs, learning_rate))
  model_fname = model_fname.replace('.', '_')
  model_dir = os.path.join(deeprad_dir, 'models', model_fname)
  if not os.path.isdir(model_dir):
    utils.make_dir_safely(model_dir)

  # Define model, loss, image fpaths
  model_fpath = os.path.join(model_dir, model_fname + '.pt')
  train_loss_fname = model_fname.replace('model', 'train_loss')
  train_loss_arr_fpath = os.path.join(model_dir, train_loss_fname + '.npy')
  train_loss_img_fpath =  os.path.join(model_dir, train_loss_fname + '.jpg')
  test_loss_fname = model_fname.replace('model', 'test_loss')
  test_loss_arr_fpath = os.path.join(model_dir, test_loss_fname + '.npy')
  test_loss_img_fpath =  os.path.join(model_dir, test_loss_fname + '.jpg')

  # Load training/test set
  dataset = CustomDataSet(TARGET_FOLDER_PATH, CHANNEL_FOLDER_PATH,
                          transform=transforms.ToTensor(), device=device)
  train_size = int(len(dataset) * .8)
  test_size = len(dataset) - train_size
  train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
  train_iters = int(train_size // batch_size)
  test_iters = int(test_size // batch_size)

  print('data size: {}: train size: {} + test size: {}'.format(len(dataset), train_size, test_size))
  print('batch size: {}'.format(batch_size))
  #print('train/test_iters = {}, {}\n'.format(train_iters, test_iters))

  if (test_size / batch_size) < 1.0:
    raise Exception('batch size {} too large for test size: {}'.format(test_size, batch_size))

  # for use in training and testing
  train_criterion = nn.MSELoss() # mean square error loss
  test_criterion = nn.MSELoss()

  #---------------------------
  # Train/Test
  #---------------------------
  print('\nTraining...')
  # if not run_train:
  #   print('Loading training model from {}.'.format(model_fname))

  # else:
    model = Autoencoder(device)
    model = model.to(device)

    print("Training {} data, over {} epochs".format(train_size, max_epochs))

    torch.manual_seed(42)
    optimizer = torch.optim.Adam(
      model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    outputs_train = [0] * max_epochs
    train_loss_arr = np.zeros((max_epochs, 1))
    start_time = time.time()

    for epoch in range(max_epochs):
      #---------------------------
      # training
      #---------------------------
      train_loss = 0
      for i, (input_batch, label_batch) in enumerate(train_loader):
          input_batch, label_batch = input_batch.to(device), label_batch.to(device)

          # forward pass
          recon_batch = model.forward(input_batch)
          # calc loss
          loss = train_criterion(recon_batch, label_batch)
          train_loss = train_loss + loss.item()

          # calc gradient with autograd
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          if i == (train_iters - 1):
            print('Train epoch:{}, Loss:{:.4f}'.format(epoch + 1, loss))
            outputs_train[epoch] = (label_batch, recon_batch)
            train_loss_arr[epoch] = train_loss / train_iters

          del input_batch; del label_batch; del recon_batch; del loss; del i

      #---------------------------
      # testing
      #---------------------------



    print('time took training {}.'.format(utils.timer(start_time, time.time())))

    # Save training model, image, losses
    torch.save(model, model_fpath)
    np.save(train_loss_arr_fpath, train_loss_arr)
    traintest_utils.viz_loss(
      outputs_train, img_fpath=train_loss_img_fpath, img_title='Train Loss', show_plot=True)
    print('Saved training model, loss image and data: {}'.format(model_fname))


  #---------------------------
  # testing
  #---------------------------
  print('\nTesting...')
  if not run_test:
    print('Loading test losses from {}.'.format(model_fname))
  else:
    print("Testing {} data, over {} epochs".format(test_size, max_epochs))

    with torch.no_grad():  # to prevent out of memory errors
      model = torch.load(model_fpath)
      model.eval()
      torch.manual_seed(42)

      test_loader = torch.utils.data.DataLoader(
          test_data, batch_size=batch_size, shuffle=True, drop_last=True)

      outputs_test = [0] * max_epochs
      test_loss_arr = np.zeros((max_epochs, 1))

      start_time = time.time()
      for epoch in range(max_epochs):
        test_loss = 0
        for i, (input_batch, label_batch) in enumerate(test_loader):
            input_batch, label_batch = input_batch.to(device), label_batch.to(device)
            recon_batch = model(input_batch)

            # calc loss
            loss = test_criterion(recon_batch, label_batch)
            test_loss = test_loss + loss.item()
            # for every epoch save data
            if i == (test_iters - 1):
                print('Test epoch:{}, Loss:{:.4f}'.format(epoch + 1, loss))
                outputs_test[epoch] = (label_batch, recon_batch)
                test_loss_arr[epoch] = test_loss / test_iters

            del input_batch; del label_batch; del recon_batch; del i


    print('time took testing {}.\n'.format(utils.timer(start_time, time.time())))

    # Save test image, losses
    np.save(test_loss_arr_fpath, test_loss_arr)
    traintest_utils.viz_loss(
      outputs_test, img_fpath=test_loss_img_fpath, img_title='Test Loss', show_plot=False)
    print('Saved testing model, loss image and data: {}'.format(model_fname))


  # Load data
  train_loss_arr = np.load(train_loss_arr_fpath)
  test_loss_arr = np.load(test_loss_arr_fpath)

  # Plot loss curves
  f, a = plt.subplots()
  a.plot(train_loss_arr, color='red', label='train')
  a.plot(test_loss_arr, color='blue', label='test')
  a.set_title('Loss for {}'.format(model_fname))
  a.set_xlabel('Epochs'); a.set_ylabel('Loss')
  a.legend(loc='upper left')
  plt.show()
  print('\nFinished\n')


if __name__ == "__main__":
  small_train = 0
  run_train, run_test = 1, 1
  max_epochs, batch_size = 30, 20

  if len(sys.argv) > 1:
    argv = sys.argv[1:]
    if '--max_epochs' in argv:
      i = argv.index('--max_epochs')
      max_epochs = int(argv[i + 1])

    if '--run_train' in argv:
      i = argv.index('--run_train')
      run_train = bool(int(argv[i + 1]))

    if '--run_test' in argv:
      i = argv.index('--run_test')
      run_test = bool(int(argv[i + 1]))

    if '--small_train' in argv:
      i = argv.index('--small_train')
      small_train = bool(int(argv[i + 1]))

    if '--batch_size' in argv:
      i = argv.index('--batch_size')
      batch_size = int(argv[i + 1])

  print('User params (e#, b#, ts, rtrn, rtst): '
    '{}, {}, {}, {}, {}'.format(max_epochs, batch_size, small_train, run_train, run_test))

  main(max_epochs=max_epochs, batch_size=batch_size,
       run_train=run_train, run_test=run_test, small_train=small_train)