import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import math
import pyimgsaliency.pyimgsaliency as psal
from scipy.io import loadmat
from PIL import Image
import cv2
from SalientEyePointGenerator import SalientEyePointGenerator
from draw_data import draw_picture_points

with open('metadata_table.npy', 'rb') as f:
    md_df = np.load(f, allow_pickle=True)

with open('picture_table.npy', 'rb') as f:
    pic_df = np.load(f, allow_pickle=True)

with open('salient_table.npy', "rb") as f:
    sal_df = np.load(f, allow_pickle=True)

with open('best_table.npy', "rb") as f:
    best_df = np.load(f, allow_pickle=True)

#HYPERPARAMETERS

#TRAINING PARAMETERS
epochs = 20
weight_decay = 1e-2
learning_rate = 0.005

net = SalientEyePointGenerator()

loss_function = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

for epoch in range(epochs):
    for i in range(md_df.shape[0]):
        if sal_df[i][0] is None:
            print("Non-valid entry found... Skipping...")
            continue
        net.zero_grad()
        out = net(md_df[i], pic_df[i], sal_df[i])
        target = best_df[i]
        target = torch.from_numpy(target)
        target = target.type(torch.FloatTensor)
        target.requires_grad_(False)
        loss = loss_function(out, target)
        loss.backward()
        optimizer.step()
        print("Epoch:{} Iter:{} Loss:{}".format(epoch + 1, i + 1, loss))
    torch.save(net.state_dict(), "/home/nvinden/ML/EyeTracking/test.model")



'''
img_np = image_pt.numpy()
cv2.imwrite("/home/nvinden/ML/EyeTracking/current_img.jpg", img_np)
'''
