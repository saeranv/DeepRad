import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from torch.utils.data import Dataset, DataLoader
from PIL import *

import os

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        # self.total_imgs = natsort.natsorted(all_imgs)
        self.total_imgs = all_imgs

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image
        # return image


img_folder_path = "data/load_test"
batch_size = 10

my_dataset = CustomDataSet(img_folder_path, transform=transforms.ToTensor())
train_loader = DataLoader(my_dataset , batch_size=batch_size, shuffle=True)
print(my_dataset[1])
