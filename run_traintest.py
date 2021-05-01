import sys
import os
import glob
import time

# torch
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

# raytune
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


# sci py stuff
import numpy as np
import matplotlib.pyplot as plt

# set path
deeprad_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
if deeprad_path not in sys.path:
  sys.path.insert(0, deeprad_path)


# Import deeprad models
from deeprad.traintest import viz_loss, training, testing
from deeprad.model import Autoencoder, CustomDataSet
from deeprad import utils
fd, pp = utils.fd, utils.pp

# Set seeds/device
np.random.seed(2)  # TODO: confirm right location?
device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
print("You are using device: %s" % device)

# Data folders
TARGET_FOLDER_PATH = os.path.join("data/traintest/out_data")
CHANNEL_FOLDER_PATH = os.path.join("data/traintest/in_data")
assert os.path.isdir(TARGET_FOLDER_PATH) and os.path.isdir(CHANNEL_FOLDER_PATH)


#---------------------------
# main
#---------------------------

torch.cuda.empty_cache()

# Hyperparameters
max_epochs = 2#15 #30
learning_rate = 1e-3
model_fpath = 'model_{}'.format('epoch_{}_lr_{}'.format(max_epochs, learning_rate))
model_fpath = model_fpath.replace('.', '_')
model_fpath = os.path.join(deeprad_path, 'deeprad', 'models', model_fpath + '.pt')

RUN_TRAIN = True

# Load training/test set
dataset = CustomDataSet(TARGET_FOLDER_PATH, CHANNEL_FOLDER_PATH,
                        transform=transforms.ToTensor(), device=device)
train_size = int(len(dataset) * .8)
test_size = len(dataset) - train_size
train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

# train_loader = DataLoader(train_data.dataset, batch_size=2, shuffle=True)
# xtrain, y = next(iter(train_loader))
# xtrain, y = xtrain.to(device), y.to(device)
# print(xtrain.is_cuda, y.is_cuda)

# training loop
if RUN_TRAIN:
  model = Autoencoder(device)
  model = model.to(device)

  start_time = time.time()
  print("Training {} data, over {} epochs".format(train_size, max_epochs))

  outputs_train = training(model, train_data, device, num_epochs=max_epochs,
                           batch_size=20, learning_rate=learning_rate)

  print('time took training {}.'.format(utils.timer(start_time, time.time())))

  # Save Model
  torch.save(model, model_fpath)

  print('Viewing Train Images')
  #viz_loss(outputs_train)

print(len(outputs_train))

# testing loop
model = torch.load(model_fpath)
model.eval()
outputs_test = testing(model, test_data, device)

print('Viewing Test Images')
viz_loss(outputs_test)
torch.cuda.empty_cache()

