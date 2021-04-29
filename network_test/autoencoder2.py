import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np

from torch.utils.data import Dataset, DataLoader
from PIL import *
# import utils

import os

import glob
import math
import random
import re


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(6, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2,
                               padding=1, output_padding=1),
            # nn.Sigmoid()
            # nn.Tanh
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class CustomDataSet(Dataset):
    def __init__(self, main_dir, ch_dir, transform):
        self.main_dir = main_dir
        self.ch_dir = ch_dir

        self.transform = transform
        all_imgs = os.listdir(main_dir)
        ch_imgs = os.listdir(ch_dir)

        all_imgs = self.sort_img(all_imgs)
        ch_imgs = self.sort_img(ch_imgs)

        # only works for 2 input channels
        half_ch_imgs = int(len(ch_imgs) / 2)
        ch_imgs = [(ch_imgs[i], ch_imgs[i+1]) for i in range(half_ch_imgs)]

        self.total_imgs = all_imgs
        self.ch_imgs = ch_imgs

    @staticmethod
    def img_fpath_idx(ss):
        """Get model id prefix from imagefpath."""
        return int(ss.split("_")[0])

    def sort_img(self, img_fpaths):
        return sorted(img_fpaths, key=lambda ss: self.img_fpath_idx(ss))

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        img_loc_c1 = os.path.join(self.ch_dir, self.ch_imgs[idx][0])
        img_loc_c2 = os.path.join(self.ch_dir, self.ch_imgs[idx][1])

        image = Image.open(img_loc).convert("RGB")
        image_c1 = Image.open(img_loc_c1).convert("RGB")
        image_c2 = Image.open(img_loc_c2).convert("RGB")
        x = torch.cat((self.transform(image_c1),
                       self.transform(image_c2)), dim=0)
        y = self.transform(image)

        return x, y

def test_model(model, batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=True)
    outputs = []

    for data in test_loader:
        img, _ = data
        recon = model(img)

        outputs.append((img, recon),)
    return outputs

def reduce_img(path, train_path, test_path):
    image_list = []
    for i, filename in enumerate(glob.glob(path + '/*.jpg')):

        if filename.lower().endswith('.jpg'):
            g = filename.split('.')[0]

            img = Image.open(filename).convert("RGB")
            new_name = filename.split('/')[1]

            img_c1 = Image.open(filename).convert("RGB")

            w, h = img.size
            # print(w,h)
            # img = Image.open(filename)
            border = 230
            x = 230
            y = 230
            w_ = 1328
            h_ = 240
            img = img.crop((x, y, x + w_, y + h_))
            new_img = img.resize((math.ceil(w * .3), int(h * .3)))
            if i % 7 == 0:
                new_img.save(test_path + str(i) + "_" +
                             new_name, optimize=True)
            else:
                new_img.save(train_path + new_name, optimize=True)

def train_model(model, num_epochs=5, batch_size=4, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss()  # mean square error loss

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)  # <--

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    outputs = []
    for epoch in range(num_epochs):
        for data in train_loader:

            img, targ = data

            recon = model.forward(img)

            loss = criterion(recon, targ)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch + 1, float(loss)))
        # outputs.append((epoch, img, recon),)
        outputs.append((epoch, targ, recon),)
    return outputs

def color2rad(img, mask=False):
    """img is np.ndarray of floats btwn 0 - 1"""
    img = (img * 255).astype(np.uint8)
    # TODO: add a mask here??
    if mask:
        img = np.where(img < (255 - 1e-10), img, np.nan)
    return img


#-------------------------------------
# reset train Data folders
#-------------------------------------

curr_dir = os.path.join(os.getcwd(), '..', 'data')
target_folder_path = os.path.join(curr_dir, "traintest/out_data")
channel_folder_path = os.path.join(curr_dir, "traintest/in_data")
print(os.path.isdir(target_folder_path), os.path.isdir(channel_folder_path))


RADCMAP = plt.get_cmap('RdYlBu_r')
np.random.seed(2)

#############################
#### Reduce Image Size ######
#############################

# Test Data Folders
# img_folder_path_test = "small_test"
# c1_folder_path_test = "small_test_c1"
# c2_folder_path_test = "small_test_c2"

train_data = CustomDataSet(target_folder_path,
                           channel_folder_path, transform=transforms.ToTensor())
#exit()
#test_data = CustomDataSet(img_folder_path_test, c1_folder_path_test,
#                          c2_folder_path_test, transform=transforms.ToTensor())


#########################################
########### Train Model #################
#########################################

model = Autoencoder()
max_epochs = 30
outputs = train_model(model, num_epochs=max_epochs)

# print(len(outputs))
for k in range(0, max_epochs, 5):
    # print(k)
    # print(outputs[k][1])
    plt.figure(figsize=(9, 2))
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 9:
            break
        plt.subplot(2, 9, i + 1)
        img = item[0]
        plt.imshow(color2rad(img), cmap=RADCMAP, vmin=0, vmax=255)
        # imgfpath = os.path.join(os.getcwd(), 'img_{}.jpg'.format(k))
        # utils.write_img(img, imgfpath)
        # print(len(imgs))

    for i, item in enumerate(recon):
        if i >= 9:
            break
        plt.subplot(2, 9, 9 + i + 1)
        img = item[0]
        img = item[0]
        plt.imshow(color2rad(img), cmap=RADCMAP, vmin=0, vmax=255)
        #imgfpath = os.path.join(os.getcwd(), 'recon_{}.jpg'.format(i))
        #utils.write_img(item[0], imgfpath)

    plt.show()
#########################################
############ Test Model #################
#########################################
print('Viewing Test Images')


# outputs_test = test_model(model)

# for k in range(0, len(outputs_test), 5):
#     plt.figure(figsize=(9, 2))
#     imgs = outputs_test[k][0].detach().numpy()
#     recon = outputs_test[k][1].detach().numpy()
#     for i, item in enumerate(imgs):
#         if i >= 9:
#             break
#         plt.subplot(2, 9, i + 1)
#         img = item[0]
#         plt.imshow(color2rad(img), cmap=RADCMAP, vmin=0, vmax=255)
#     for i, item in enumerate(recon):
#         if i >= 9:
#             break
#         plt.subplot(2, 9, 9 + i + 1)
#         img = item[0]
#         plt.imshow(color2rad(img), cmap=RADCMAP, vmin=0, vmax=255)

#     plt.show()
