import os
import numpy as np

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

# Image
from PIL import Image

# set seed
np.random.seed(2)

class Autoencoder(nn.Module):
    def __init__(self, device, f1=16, k1=3):
        super(Autoencoder, self).__init__()
        self.device = device
        f2 = f1 * 2
        f3 = f2 * 2
        f4 = f3 * 2
        k2 = k1 * 2

        # TODO: add nonlinear layer?
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(6, f1, k1, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(f1, f2, k1, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(f2, f3, k2),
            nn.LeakyReLU(),
            nn.Conv2d(f3, f4, k2)
        )

        # TODO: add softmax?
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(f4, f3, k2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(f3, f2, k2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(f2, f1, k1, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(f1, 3, k1, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
            #nn.LeakyReLU()#,nn.Tanh
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class CustomDataSet(Dataset):
    def __init__(self, main_dir, ch_dir, transform, device):
        self.main_dir = main_dir
        self.ch_dir = ch_dir

        self.transform = transform
        all_imgs = os.listdir(main_dir)
        ch_imgs = os.listdir(ch_dir)

        all_imgs = self.sort_img(all_imgs)
        ch_imgs = self.sort_img(ch_imgs)

        # split channels from flat list
        ch_1, ch_2 = self.split_channels(ch_imgs)

        self.total_imgs = all_imgs
        self.ch_1 = ch_1
        self.ch_2 = ch_2

        self.device = device

    @staticmethod
    def img_fpath_idx(ss):
        """Get model id prefix from imagefpath."""
        return int(ss.split("_")[0])

    @staticmethod
    def sort_img(img_fpaths):
        return sorted(img_fpaths, key=lambda ss: CustomDataSet.img_fpath_idx(ss))

    def split_channels(self, ch_imgs):
        c1, c2 = [], []
        channels = iter(ch_imgs)
        for x in channels:
            c1.append(x)
            c2.append(next(channels))
        return c1, c2

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        """Load image from directory given index."""
        crop_x, crop_y = 0, 9
        crop_w, crop_h = 1300, 108

        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        ch1_loc = os.path.join(self.ch_dir, self.ch_1[idx])
        ch2_loc = os.path.join(self.ch_dir, self.ch_2[idx])

        # Load labels/input channels
        image = Image.open(img_loc).convert("RGB")
        channel_1 = Image.open(ch1_loc).convert("RGB")
        channel_2 = Image.open(ch2_loc).convert("RGB")

        # Crop inputs
        image = image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        channel_1 = channel_1.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        channel_2 = channel_2.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))

        # crop
        # image = np.array(image)[crop_x: crop_w, crop_y : crop_h]
        # channel_1 = np.array(channel_1)[crop_x: crop_w, crop_y: crop_h]
        # channel_2 = np.array(channel_2)[crop_x: crop_w, crop_y: crop_h]

        x_in = torch.cat((self.transform(channel_1), self.transform(channel_2)), dim=0)
        y_lbl = self.transform(image)

        #x_in = x_in.int()
        #y_lbl = y_lbl.int()

        del image; del channel_1; del channel_2; del img_loc; del ch1_loc; del ch2_loc

        # import matplotlib.pyplot as plt
        # y_lbl = y_lbl.permute(1, 2, 0)
        # print(y_lbl.size())
        # plt.imshow(y_lbl)
        # plt.show()
        # assert False

        return x_in, y_lbl