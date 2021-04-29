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
import time


RADCMAP = plt.get_cmap('RdYlBu_r')

np.random.seed(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device: %s" % device)

#############################
#### Reduce Image Size ######
#############################


img_folder_path = "small_train"
org_folder_path = "load_test"
org_folder_pathc1 = "load_test_c1"
org_folder_pathc2 = "load_test_c2"


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


# reduce_img(org_folder_path, 'small_train/', 'small_test/')
# reduce_img(org_folder_pathc1, 'small_train_c1/', 'small_test_c1/')
# reduce_img(org_folder_pathc2, 'small_train_c2/', 'small_test_c2/')
# exit()

#############################
#############################


class Autoencoder(nn.Module):
    def __init__(self, device=None):
        super(Autoencoder, self).__init__()
        self.device = device
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


#############################
###### Load Custom Data #####
#############################

class CustomDataSet(Dataset):
    def __init__(self, main_dir, ch_dir, transform):
        self.main_dir = main_dir
        self.ch_dir = ch_dir
        # self.c2_dir = c2_dir

        self.transform = transform
        all_imgs = os.listdir(main_dir)
        ch_imgs = os.listdir(ch_dir)
        # c2_imgs = os.listdir(c2_dir)

        all_imgs = self.sort_img(all_imgs)
        ch_imgs = self.sort_img(ch_imgs)

        ch_1, ch_2 = self.split_channels(ch_imgs)


        self.total_imgs = all_imgs
        self.ch_1 = ch_1
        self.ch_2 = ch_2


    def sort_img(self, l):
        keys = []
        names = []
        for j, i in enumerate(l):
            k = i.split('_')[0]
            keys.append(int(k))
            names.append(i)

        _, sorted_list = (list(t) for t in zip(*sorted(zip(keys, names))))
        # print(sorted_list[:5])

        return sorted_list
    
    def split_channels(self, l):
        c1, c2 = [], []
        channels = iter(l)
        for x in channels:
            c1.append(x)
            c2.append(next(channels))
        
        return c1, c2

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
       
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        ch1_loc = os.path.join(self.ch_dir, self.ch_1[idx])
        ch2_loc = os.path.join(self.ch_dir, self.ch_2[idx])

        image = Image.open(img_loc).convert("RGB")
        channel_1 = Image.open(ch1_loc).convert("RGB")
        channel_2 = Image.open(ch2_loc).convert("RGB")

        x,y = 0, 0
        w_ = 1300
        h_ = 108
        image = image.crop((x, y, x + w_, y + h_))
        channel_1 = channel_1.crop((x, y, x + w_, y + h_))
        channel_2 = channel_2.crop((x, y, x + w_, y + h_))

        ### Resize Img
        # factor = .52
        # image = image.resize((math.ceil(w_ * factor), int(h_ * factor)))
        # channel_1 = channel_1.resize((math.ceil(w_ * factor), int(h_ * factor)))
        # channel_2 = channel_2.resize((math.ceil(w_ * factor), int(h_ * factor)))



        x = torch.cat((self.transform(channel_1),
                       self.transform(channel_2)), dim=0)

        y = self.transform(image)
        return x, y


# Train Data folders
target_folder_path = "data/traintest/out_data"
channel_folder_path = "data/traintest/in_data"

# Test Data Folders
# img_folder_path_test = "small_test"
# c1_folder_path_test = "small_test_c1"
# c2_folder_path_test = "small_test_c2"

dataset = CustomDataSet(target_folder_path,
                           channel_folder_path, transform=transforms.ToTensor())
train_size = int(len(dataset) * .8)
test_size = len(dataset) - train_size
train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size]) 

#########################################
########### Train Model #################
#########################################


def train(model, num_epochs=5, batch_size=40, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss()  # mean square error loss

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)  # <--

    train_loader = DataLoader(train_data.dataset, batch_size=batch_size, shuffle=True)

    outputs = []
    for epoch in range(num_epochs):
        # start = time.time()

        for data in train_loader:

            img, targ = data

            recon = model.forward(img)
            loss = criterion(recon, targ)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # now = time.time()
        # elapsed = (now - start)/60
        print('Epoch:{}, Loss:{:.4f}'.format(epoch + 1, float(loss)))
        # print('Epoch:{}, Loss:{:.4f}, Time: {:.4f}'.format(epoch + 1, float(loss), elapsed))
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



model = Autoencoder(device)
max_epochs = 20

print("Training")
outputs = train(model, num_epochs=max_epochs, learning_rate= 1e-3)

# print(len(outputs))
grid=2
for k in range(0, max_epochs, 5):
    # print(k)
    # print(outputs[k][1])
    plt.figure(figsize=(grid, 2))
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()
    for i, item in enumerate(imgs):
        if i >= grid:
            break
        plt.subplot(2, grid, i + 1)
        img = item[0]
        plt.imshow(color2rad(img), cmap=RADCMAP, vmin=0, vmax=255)
        # imgfpath = os.path.join(os.getcwd(), 'img_{}.jpg'.format(k))
        # utils.write_img(img, imgfpath)
        # print(len(imgs))

    for i, item in enumerate(recon):
        if i >= grid:
            break
        plt.subplot(2, grid, grid + i + 1)
        img = item[0]
        img = item[0]
        plt.imshow(color2rad(img), cmap=RADCMAP, vmin=0, vmax=255)
        #imgfpath = os.path.join(os.getcwd(), 'recon_{}.jpg'.format(i))
        #utils.write_img(item[0], imgfpath)

    plt.show()

# Save Model
torch.save(model, 'model.pt')

# Load
# model = torch.load(PATH)
# model.eval()
exit()
#########################################
############ Test Model #################
#########################################
print('Viewing Test Images')


def test(model, batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)

    test_loader = torch.utils.data.DataLoader(test_data.dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    outputs = []

    for data in test_loader:
        img, _ = data
        recon = model(img)

        outputs.append((img, recon),)
    return outputs


outputs_test = test(model)

for k in range(0, len(outputs_test), 5):
    plt.figure(figsize=(9, 2))
    imgs = outputs_test[k][0].detach().numpy()
    recon = outputs_test[k][1].detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 9:
            break
        plt.subplot(2, 9, i + 1)
        img = item[0]
        plt.imshow(color2rad(img), cmap=RADCMAP, vmin=0, vmax=255)
    for i, item in enumerate(recon):
        if i >= 9:
            break
        plt.subplot(2, 9, 9 + i + 1)
        img = item[0]
        plt.imshow(color2rad(img), cmap=RADCMAP, vmin=0, vmax=255)

    plt.show()
