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
        f3 = f1 * 4
        f4 = f1 * 6  # 128
        f5 = f1 * 8
        f6 = f1 * 10
        k2 = k1 * 2

        # TODO: add nonlinear layer?
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(6, f1, k1, stride=2, padding=1),
            nn.BatchNorm2d(num_features=f1),
            nn.LeakyReLU(),
            nn.Conv2d(f1, f2, k1, stride=2, padding=1),
            nn.BatchNorm2d(num_features=f2),
            nn.LeakyReLU(),
            nn.Conv2d(f2, f3, k2),
            nn.BatchNorm2d(num_features=f3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=f3, out_channels=f4, kernel_size=k2),
            nn.BatchNorm2d(num_features=f4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=f4, out_channels=f5 , kernel_size=k2),
            nn.BatchNorm2d(num_features=f5),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=f5, out_channels=f6 , kernel_size=k2),
            nn.BatchNorm2d(num_features=f6),
            nn.LeakyReLU()
        )

        # in_features = 1542240
        # hidden_dim = 1000
        # self.fc1 = nn.Linear(in_features=in_features,
        #                      out_features=hidden_dim, bias=True)
        # self.fc2 = nn.Linear(in_features=hidden_dim,
        #                      out_features=in_features, bias=True)

        # TODO: add softmax?
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(f6, f5, k2),
            nn.BatchNorm2d(num_features=f5),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(f5, f4, k2),
            nn.BatchNorm2d(num_features=f4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(f4, f3, k2),
            nn.BatchNorm2d(num_features=f3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(f3, f2, k2),
            nn.BatchNorm2d(num_features=f2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(f2, f1, k1, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=f1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(f1, 3, k1, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        # def _flatten_x(_x):
        #     """Flatten for fully connected layer."""
        #     flat_size = 1
        #     for dim in _x.shape[1:]:
        #         flat_size *= dim
        #     return _x.view(-1, flat_size)

        x = self.encoder(x)
        #x = _flatten_x(x)
        #x = torch.flatten(x)
        #print('chk flatten: ', x.shape)
        #x = self.fc1(x)
        #x = self.fc2(x)
        x = self.decoder(x)
        return x

class CustomDataSet(Dataset):
    def __init__(self, main_dir, ch_dir, transform, device, data_limit=None):
        self.main_dir = main_dir
        self.ch_dir = ch_dir

        self.transform = transform
        all_imgs = os.listdir(main_dir)
        ch_imgs = os.listdir(ch_dir)

        if data_limit:
            all_imgs = all_imgs[:data_limit + 1]
            ch_imgs = ch_imgs[:data_limit + 1]

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
        len_ch = len(ch_imgs) // 2
        c1, c2 = [0] * len_ch , [0] * len_ch
        for i in range(len_ch):
            c1[i] = ch_imgs[i]
            c2[i] = ch_imgs[i + 1]
        return c1, c2

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        """Load image from directory given index."""
        crop_x, crop_y = 0, 9
        crop_w, crop_h = 1300, 108

        print(idx)
        print(self.total_imgs)

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