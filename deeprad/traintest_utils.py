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

    if show_plot:
        plt.show()


def plot_curves(train_loss_history, train_acc_history, valid_loss_history, valid_acc_history, hidx=None):
    '''
    Code referenced from author (Saeran Vasanthakumar's) learning curve implementation from assignment 1.

    Plot learning curves with matplotlib. Make sure training loss and validation loss are plot in the same figure and
    training accuracy and validation accuracy are plot in the same figure too.
    :param train_loss_history: training loss history of epochs
    :param train_acc_history: training accuracy history of epochs
    :param valid_loss_history: validation loss history of epochs
    :param valid_acc_history: validation accuracy history of epochs
    :return: None, save two figures in the current directory
    '''
    #############################################################################
    # TODO:                                                                     #
    #    1) Plot learning curves of training and validation loss                #
    #    2) Plot learning curves of training and validation accuracy            #
    #############################################################################

    train_loss = train_loss_history
    train_acc = train_acc_history
    valid_loss = valid_loss_history
    valid_acc = valid_acc_history

    suptitle = 'Learning Curves'

    # plot
    epochs = train_loss.shape[0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=False)
    fig.subplots_adjust(wspace=0.27, hspace=None)
    fig.suptitle(suptitle, fontsize=12, y=1)

    # loss
    ax[0].plot(np.arange(epochs), train_loss, 'blue')
    ax[0].plot(np.arange(epochs), valid_loss, 'red')
    ax[0].set_title('Loss over Epochs')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    _ = ax[0].set_xticks(np.arange(epochs))
    ax[0].grid()

    # accuracy
    ax[1].plot(np.arange(epochs), train_acc, 'blue')
    ax[1].plot(np.arange(epochs), valid_acc, 'red')
    ax[1].set_title('Accuracy over Epochs')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    _ = ax[1].set_xticks(np.arange(epochs))
    ax[1].grid()

    _ = fig.gca().legend(('training history', 'validation history'),
                         bbox_to_anchor=(-0.745, -0.2))

    # Manually adjust to specify model if desired
    #model_type = 'unknown_model'
    model_type = '2_layer_nn'
    #model_type = 'softmax_reg'


    # TODO: temp
    import os
    #hparam_name = 'hiddens'
    hparam_name = 'regs'
    folder = os.path.join(os.getcwd(), '_final_data', hparam_name)
    #params_fpath = os.path.join(folder, '_{}_params.npy'.format(model_type))
    #params = np.load(os.path.join(folder, '_{}_model_params.npy'.format(model_type)))

    #fname = os.path.join(folder, '_{}_learning_curve_hi_{}.png'.format(model_type, hidx))
    #plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w', pad_inches=0.1)
    #print('plotted at ./{}.png'.format(fname))