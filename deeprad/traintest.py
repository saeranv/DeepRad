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

def viz_loss(outputs):
    grid = 2
    num_epochs = len(outputs)
    for k in range(num_epochs, 5):
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

def training(model, train_data, device, num_epochs=5, batch_size=40, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss()  # mean square error loss

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)  # <--

    train_loader = DataLoader(train_data.dataset, batch_size=batch_size, shuffle=True)

    outputs = []
    for epoch in range(num_epochs):
        # start = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            recon = model.forward(inputs)
            loss = criterion(recon, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # now = time.time()
        # elapsed = (now - start)/60
        print('Epoch:{}, Loss:{:.4f}'.format(epoch + 1, float(loss)))
        # print('Epoch:{}, Loss:{:.4f}, Time: {:.4f}'.format(epoch + 1, float(loss), elapsed))
        # outputs.append((epoch, img, recon),)
        outputs.append((epoch, labels, recon),)
    return outputs

def testing(model, test_data, device, batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)

    test_loader = torch.utils.data.DataLoader(test_data.dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    outputs = []

    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        recon = model(inputs)
        outputs.append((inputs, recon),)

    return outputs

