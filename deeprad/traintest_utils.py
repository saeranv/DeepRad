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


def viz_loss(outputs, img_fpath, img_title='Radiation Map', xdim=10):

    assert xdim % 10 == 0  # multiple of 10
    data_num = len(outputs)
    f, a = plt.subplots(data_num, 2, figsize=(xdim, ((xdim // 10) * data_num) + 5))
    f.subplots_adjust(top=0.9)
    for i, (label_batch, recon_batch) in enumerate(outputs):
        # not point in plotting all in batch so just select first
        label = label_batch[0].permute(1, 2, 0).detach().cpu().numpy()
        recon = recon_batch[0].permute(1, 2, 0).detach().cpu().numpy()

        label = clean_img4rad(label) / 255.0 * 1000.0
        recon = clean_img4rad(recon) / 255.0 * 1000.0

        im = a[i, 0].imshow(label, cmap=RADCMAP, vmin=0, vmax=1000)
        im = a[i, 1].imshow(recon, cmap=RADCMAP, vmin=0, vmax=1000)
        clean_plot4rad(a[i], i)

    # left, bottom, width, height
    cax = f.add_axes([0.125, 0.08, 0.1, 0.025])
    f.colorbar(im, orientation='horizontal', cax=cax)

    st = f.suptitle(img_title)
    st.set_y(0.97)

    plt.savefig(img_fpath)