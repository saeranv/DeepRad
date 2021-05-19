import os
import sys
import time
import gc

# sci py stuff
import matplotlib.pyplot as plt
import numpy as np
import PIL

# torch
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

gc.collect()
torch.cuda.empty_cache()

try:
    from ray import tune
    #from ray.tune import CLIReporter
    #from ray.tune.schedulers import ASHAScheduler
    #from ray.tune.suggest.hyperopt import HyperOptSearch
except:
    print('skip ray install')

# set path
deeprad_dir = os.path.abspath(os.path.join(os.getcwd()))
if deeprad_dir not in sys.path:
  sys.path.insert(0, deeprad_dir)

# Import deeprad models
from deeprad.model import CustomDataSet
from deeprad import utils
import deeprad.traintest_utils as traintest_utils
fd, pp = utils.fd, utils.pp


RADCMAP = plt.get_cmap('RdYlBu_r')

def rgb2gray(rgb):
    return np.dot(rgb, [0.2989, 0.5870, 0.1140])

def clean_img4rad(img, mask=False):
    """img is np.ndarray of floats btwn 0 - 1"""
    img_delta = img.max() - img.min()
    img = (img - img.min())/ img_delta
    img = img * 255
    img = img.astype(np.uint8)
    img = rgb2gray(img)

    # TODO: add a mask here??
    #if mask:
    #    img = np.where(img < (255 - 1e-10), img, np.nan)

    return img

def get_traintest_fpaths(deeprad_dir, model_dir, model_fname):
    """Make fpaths from parent dirs."""
    model_fpath = os.path.join(model_dir, model_fname + '.pt')
    train_loss_fname = model_fname.replace('model', 'train_loss')
    train_loss_arr_fpath = os.path.join(model_dir, train_loss_fname + '.npy')
    train_loss_img_fpath =  os.path.join(model_dir, train_loss_fname + '.jpg')
    test_loss_fname = model_fname.replace('model', 'test_loss')
    test_loss_arr_fpath = os.path.join(model_dir, test_loss_fname + '.npy')
    test_loss_img_fpath =  os.path.join(model_dir, test_loss_fname + '.jpg')
    learning_loss_img_fpath = os.path.join(model_dir, 'learning_curve.jpg')

    return model_fpath, train_loss_fname, train_loss_arr_fpath, train_loss_img_fpath, \
      test_loss_fname, test_loss_arr_fpath, test_loss_img_fpath, learning_loss_img_fpath


def viz_learning_curve(train_loss_arr, test_loss_arr, model_fname, img_fpath, a=None, **kwargs):
    """Plot learning curves for loss."""
    if not a:
        f, a = plt.subplots()

    a.plot(train_loss_arr, color='red', label='train', **kwargs)
    a.plot(test_loss_arr, color='blue', label='test', **kwargs)
    a.set_title('Loss for \n{}'.format(model_fname.replace('_', ' ')))
    a.set_xlabel('Epochs'); a.set_ylabel('Loss')
    a.legend(loc='upper left')
    plt.savefig(img_fpath)

    return a


def clean_plot4rad(axlst, epoch):
    """Mutates given list of axes."""
    for i, ax in enumerate(axlst):
        ax.axis('equal')
        ax.axis('off')
        ax.set_title('Epoch: {}'.format(epoch + 1), fontsize=7, loc='left')


def viz_loss(outputs, img_fpath, img_title='Radiation Map', xdim=20, show_plot=False):

    assert xdim % 10 == 0  # multiple of 10
    data_num = len(outputs)
    f, a = plt.subplots(data_num, 3, figsize=(xdim, ((xdim // 10) * data_num) + 5))
    #f.subplots_adjust(top=0.9)

    for i, (input_batch, label_batch, recon_batch) in enumerate(outputs):
        # not point in plotting all in batch so just select first
        _input = input_batch[:, :, :3]
        label = label_batch
        recon = recon_batch

        #recon = np.clip(recon, 0.0, 255.0).astype(int)
        #label = clean_img4rad(label) / 255.0 * 1000.0
        #recon = clean_img4rad(recon) / 255.0 * 1000.0

        im = a[i, 0].imshow(_input)
        im = a[i, 1].imshow(label)#, cmap=RADCMAP, vmin=0, vmax=1000)
        im = a[i, 2].imshow(recon)#, cmap=RADCMAP, vmin=0, vmax=1000)
        clean_plot4rad(a[i], i)

    # left, bottom, width, height
    #cax = f.add_axes([0.125, 0.08, 0.1, 0.025])
    #f.colorbar(im, orientation='horizontal', cax=cax)

    st = f.suptitle(img_title)
    #st.set_y(0.97)

    plt.savefig(img_fpath)

    if show_plot:
        plt.show()

def main(hparam, model_fname, Autoencoder, checkpoint_dir=None, in_dir=None,
         out_dir=None, run_hparam=False, data_limit=False, run_gpu=True, pretrained_model_fpath=None):

  # Set seeds/device
  np.random.seed(2)  # TODO: confirm right location?
  device = torch.device("cuda:0") if torch.cuda.is_available() and run_gpu else 'cpu'
  print("You are using device: %s" % device)
  print('hparam:')
  [print('\t{}: {}'.format(k, v)) for k, v in hparam.items()]

  # Extract epoch, batch_size for data calcs
  max_epochs, batch_size = hparam['max_epochs'], hparam['batch_size']


  # ---------------------------------------------------------------------------------
  # Define directories
  # ---------------------------------------------------------------------------------

  # for opt
  checkpoint_dir = os.path.join(deeprad_dir, 'models', 'checkpoint')
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
  dataset = CustomDataSet(out_dir, in_dir,
                          transform=transforms.ToTensor(), device=device, data_limit=data_limit)
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

  #---------------------------------------------------------------------------------
  # Epoch iteration
  #---------------------------------------------------------------------------------
  print('\nTraining/Testing...')
  print("Training/Testing {} data, over {} epochs".format(train_size, max_epochs))

  # Construct init model
  torch.manual_seed(42)

  if not pretrained_model_fpath:
    model = Autoencoder(device, hparam['f1'], hparam['k1'])
  else:
    assert os.path.isfile(pretrained_model_fpath), pretrained_model_fpath
    print('Loading pretrained model from {}'.format(pretrained_model_fpath))
    model = torch.load(pretrained_model_fpath)


  if torch.cuda.device_count() > 1:
      model = nn.DataParallel(model)
  model = model.to(device)

  # Freeze encoder
  # model.encoder.requires_grad = False

  # optimizer
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
    lr=hparam['learning_rate'], weight_decay=hparam['weight_decay'])

  # Training init
  train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, drop_last=True)
  outputs_train = [0] * max_epochs
  train_loss_arr = np.zeros((max_epochs, 1))

  # Testing init
  test_loader = DataLoader(
    test_data, batch_size=batch_size, shuffle=True, drop_last=True)
  outputs_test = [0] * max_epochs
  test_loss_arr = np.zeros((max_epochs, 1))

  #---------------------------
  # Start epoch iterations
  #---------------------------
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
        train_loss = train_loss + (loss.item() / batch_size)
        train_steps = train_steps + 1

        # calc gradient with autograd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i < (train_iters - 1):
          del input_batch; del label_batch; del recon_batch; del loss; del i
    print('Train epoch:{}, Loss:{:.4f}'.format(epoch + 1, loss))
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
        test_loss = test_loss + loss.item() / batch_size
        test_steps = test_steps + 1

        if i < (test_iters - 1):
          del input_batch; del label_batch; del recon_batch; del loss; del i

    print('Test epoch:{}, Loss:{:.4f}'.format(epoch + 1, loss))
    inter_time = utils.timer(start_time, time.time())
    print('time: {}.'.format(inter_time))
    outputs_test[epoch] = (
      input_batch[0].permute(1, 2, 0).detach().cpu().numpy(),
      label_batch[0].permute(1, 2, 0).detach().cpu().numpy(),
      recon_batch[0].permute(1, 2, 0).detach().cpu().numpy())
    test_loss_arr[epoch] = test_loss / test_steps

  #---------------------------
  # End epoch iterations
  #---------------------------

  if run_hparam:
    tune.track.log(test_loss=test_loss_arr[-1])
    torch.save(model.state_dict(), model_fpath)

  end_time = utils.timer(start_time, time.time())
  print('Time took training/testing {}.'.format(end_time))
  with open(os.path.join(model_dir, 'hparams.txt'), 'a') as hparam_file:
    hparam_file.write('last test_loss: {}\n'.format(test_loss_arr[-1]))
    hparam_file.write('test_loss: {}\n'.format(test_loss_arr))
    hparam_file.write('training time: {}\n'.format(end_time))

  out_fpaths = (model_fpath, test_loss_img_fpath, learning_loss_img_fpath)
  return model, outputs_test, train_loss_arr, test_loss_arr, out_fpaths
