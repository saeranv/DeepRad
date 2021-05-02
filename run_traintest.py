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


def main(max_epochs=30, run_train=True, train_small=False):

  # Data folders
  TARGET_FOLDER_PATH = os.path.join(deeprad_dir, "data", "traintest", "out_data")
  CHANNEL_FOLDER_PATH = os.path.join(deeprad_dir, "data", "traintest", "in_data")
  assert os.path.isdir(TARGET_FOLDER_PATH) and os.path.isdir(CHANNEL_FOLDER_PATH)


  # Hyperparameters
  max_epochs = max_epochs
  learning_rate = 1e-3
  weight_decay = 1e-5
  batch_size = 20 # 5 images trained at once in layer
  # TODO: SGD/Adam/error

  if train_small:
    TARGET_FOLDER_PATH = TARGET_FOLDER_PATH.replace('traintest', 'traintest_small')
    CHANNEL_FOLDER_PATH = CHANNEL_FOLDER_PATH.replace('traintest', 'traintest_small')

  # Model
  model_fpath = 'model_{}'.format('epoch_{}_lr_{}'.format(max_epochs, learning_rate))
  model_fpath = model_fpath.replace('.', '_')
  model_fpath = os.path.join(deeprad_dir, 'models', model_fpath + '.pt')

  # Load training/test set
  dataset = CustomDataSet(TARGET_FOLDER_PATH, CHANNEL_FOLDER_PATH,
                          transform=transforms.ToTensor(), device=device)
  train_size = int(len(dataset) * .8)
  test_size = len(dataset) - train_size
  train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
  train_iters = int(train_size // batch_size)
  test_iters = int(test_size // batch_size)

  print('\ndata size: {}, train size: {}, test size: {}'.format(len(dataset), train_size, test_size))
  print('batch size: {}'.format(batch_size))
  print('train/test_iters = {}, {}\n'.format(train_iters, test_iters))

  # for use in training and testing
  train_criterion = nn.MSELoss() # mean square error loss
  test_criterion = nn.MSELoss()

  #---------------------------
  # training
  #---------------------------
  if run_train:
    model = Autoencoder(device)
    model = model.to(device)

    start_time = time.time()
    print("Training {} data, over {} epochs".format(train_size, max_epochs))

    torch.manual_seed(42)
    optimizer = torch.optim.Adam(
      model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    outputs_train = [0] * max_epochs
    train_loss_arr = []

    for epoch in range(max_epochs):
        start = time.time()
        for i, (input_batch, label_batch) in enumerate(train_loader):
            input_batch, label_batch = input_batch.to(device), label_batch.to(device)

            # forward pass
            recon_batch = model.forward(input_batch)
            # calc loss
            loss = train_criterion(recon_batch, label_batch)

            # calc gradient with autograd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i == (train_iters - 1):
              print('Epoch:{}, Loss:{:.4f}'.format(epoch + 1, loss))
              outputs_train[epoch] = (label_batch, recon_batch)

            # For learning curves
            train_loss_arr.append(loss)
            del input_batch; del label_batch; del recon_batch; del loss; del i

    print('time took training {}.'.format(utils.timer(start_time, time.time())))

    # Save Model
    torch.save(model, model_fpath)

  #---------------------------
  # testing
  #---------------------------
  with torch.no_grad():  # to prevent out of memory errors
    model = torch.load(model_fpath)
    model.eval()
    torch.manual_seed(42)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True, drop_last=True)

    outputs_test = [0] * test_iters
    test_loss_arr = []
    for i, (input_batch, label_batch) in enumerate(test_loader):
        input_batch, label_batch = input_batch.to(device), label_batch.to(device)
        recon_batch = model(input_batch)
        loss = test_criterion(recon_batch, label_batch)
        test_loss_arr.append(loss)
        outputs_test[i] = (label_batch, recon_batch)

    # Viz loss
    _outputs_test = outputs_test[::20]  # 20
    if train_small:
      _outputs_test = outputs_test

    assert len(_outputs_test) > 1, 'outputs_test is < 1'
    loss_img_fpath =  model_fpath.replace('.pt', '.jpg').replace('model_', 'test_loss_')
    traintest_utils.viz_loss(_outputs_test, img_fpath=loss_img_fpath, img_title='Test Loss')
    print('Saved img at: {}'.format(loss_img_fpath))


# Plot loss curves
#f, a = plt.subplots()
#a.plot(train_loss_arr, color='red')
#a.plot(test_loss_arr, color='blue')



if __name__ == "__main__":
  max_epochs, run_train, train_small = 30, True, False

  if len(sys.argv) > 1:
    argv = sys.argv[1:]
    if '--max_epochs' in argv:
      i = argv.index('--max_epochs')
      max_epochs = int(argv[i + 1])

    if '--run_train' in argv:
      i = argv.index('--run_train')
      run_train = bool(int(argv[i + 1]))

    if '--train_small' in argv:
      i = argv.index('--train_small')
      train_small = bool(int(argv[i + 1]))

  print('User params (e#, rt, ts): ', max_epochs, run_train, train_small)

  main(max_epochs=max_epochs, run_train=run_train, train_small=train_small)