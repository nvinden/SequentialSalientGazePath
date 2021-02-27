import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as vision
import torchvision.transforms as tf
import pandas as pd
import numpy as np
import math
import scipy.io
from PIL import Image
import cv2
from model import HeatMapGenerator 
from EyeTrackingDataset import EyeTrackingDataset
import faulthandler

faulthandler.enable()

#Hyperparameters
mini_batch_size = 10
learning_rate = 0.05
weight_decay = 0.001
num_epochs = 50

#Batch parameters
directory_list = ["Action", "Affective", "Art", "Indoor", "Inverted", "LowResolution", "Noisy", "Object", "OutdoorManMade", "OutdoorNatural", "Social"]
root_directory = "/home/nvinden/ML/EyeTracking2/trainSet"

stimuli_dataset = EyeTrackingDataset(root_directory, directory_list)

stimuli = torch.utils.data.DataLoader(stimuli_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True)

net = HeatMapGenerator(mini_batch_size)
net.load_state_dict(torch.load("/home/nvinden/ML/EyeTracking2/test.model"))

loss_function = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

for epoch in range(num_epochs):
    for i_batch,sample_batched in enumerate(stimuli, 0):
        out = net(sample_batched["image"], sample_batched["salient"])
        target = F.adaptive_avg_pool2d(sample_batched["target"], (54, 96))
        loss = loss_function(out, target)
        loss.backward()
        optimizer.step()
        print("Epoch:{} Iteration:{} Loss:{}".format(epoch + 1, i_batch + 1, loss))
    torch.save(net.state_dict(), "/home/nvinden/ML/EyeTracking2/test.model")
