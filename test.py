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

#model parameters
mini_batch_size = 1

directory_list = ["Action"]
root_directory = "/home/nvinden/ML/EyeTracking2/Stimuli"

stimuli_dataset = EyeTrackingDataset(root_directory, directory_list, main_directory="", phase="Test")
stimuli = torch.utils.data.DataLoader(stimuli_dataset, batch_size=mini_batch_size, shuffle=False)

model = HeatMapGenerator(mini_batch_size)
model.load_state_dict(torch.load("/home/nvinden/ML/EyeTracking2/test.model"))

for i_batch,sample_batched in enumerate(stimuli, 0):
    out = model(sample_batched["image"], sample_batched["salient"], train=False)
    out = out.detach()
    print(out.shape)
    out = torch.squeeze(out, dim=0)
    out = out.permute(1, 2, 0)
    out = out.numpy()
    out = out*255
    cv2.imwrite("/home/nvinden/ML/EyeTracking2/Images/out_" + str(i_batch) + ".jpg", out*255)

