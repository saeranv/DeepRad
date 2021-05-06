import os
import glob
import time
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

# images
import matplotlib.pyplot as plt
import PIL

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

def clean_plot4rad(axlst, epoch):
    """Mutates given list of axes."""
    for i, ax in enumerate(axlst):
        ax.axis('equal')
        ax.axis('off')
        ax.set_title('Epoch: {}'.format(epoch + 1), fontsize=7, loc='left')


def viz_loss(outputs, img_fpath, img_title='Radiation Map', xdim=10, show_plot=False):

    assert xdim % 10 == 0  # multiple of 10
    data_num = len(outputs)
    f, a = plt.subplots(data_num, 2, figsize=(xdim, ((xdim // 10) * data_num) + 5))
    f.subplots_adjust(top=0.9)
    for i, (label_batch, recon_batch) in enumerate(outputs):
        # not point in plotting all in batch so just select first
        label = label_batch[0].permute(1, 2, 0).detach().cpu().numpy()
        recon = recon_batch[0].permute(1, 2, 0).detach().cpu().numpy()

        #recon = np.clip(recon, 0.0, 255.0).astype(int)
        #label = clean_img4rad(label) / 255.0 * 1000.0
        #recon = clean_img4rad(recon) / 255.0 * 1000.0

        im = a[i, 0].imshow(label)#, cmap=RADCMAP, vmin=0, vmax=1000)
        im = a[i, 1].imshow(recon)#, cmap=RADCMAP, vmin=0, vmax=1000)
        clean_plot4rad(a[i], i)

    # left, bottom, width, height
    cax = f.add_axes([0.125, 0.08, 0.1, 0.025])
    f.colorbar(im, orientation='horizontal', cax=cax)

    st = f.suptitle(img_title)
    st.set_y(0.97)

    plt.savefig(img_fpath)

    if show_plot:
        plt.show()

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