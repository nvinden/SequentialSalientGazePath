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

class SalientEyePointGenerator(nn.Module):
    def __init__(self):
        super(SalientEyePointGenerator, self).__init__()
        
        #Image Convolution
        self.img_adaptive_pool = nn.AdaptiveAvgPool2d((227, 227))
        self.convolutions = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.sal_adaptive_pool = nn.AdaptiveAvgPool2d((75, 75))
        self.lstm = nn.LSTM(9216 + 5625, 1000)
        self.linear1 = nn.Linear(1000, 256)
        self.linear2 = nn.Linear(256, 4)

    
    def forward(self, metadata, image, salience, mode="train"):
        #image parameters
        image_width = metadata[1]
        image_height = metadata[2]

        #preparing image data
        image = np.reshape(image, (image_width, image_height, -1))
        image = torch.from_numpy(image)
        image = torch.flip(image, dims=[2])
        image = image.type(torch.FloatTensor)
        image.requires_grad_(False)

        #preparing salience data
        salience = self.create_salient_image(metadata)
        salience = salience.type(torch.FloatTensor)
        salience.requires_grad_(False)

        #image and salience Adaptive Average Pooling
        img_changed = image.permute(2, 0, 1)
        img_out = self.img_adaptive_pool(img_changed)
        img_out = img_out.permute(1, 2, 0)

        sal_changed = salience.view(1,image_width,image_height)
        sal_out = self.img_adaptive_pool(sal_changed)
        sal_out = sal_out.permute(1, 2, 0)

        #concatenating image and salience
        out = torch.cat((img_out, sal_out), dim=2)

        #convolutions
        out = out.permute(2, 0, 1)
        out = out.unsqueeze(0)
        out = self.convolutions(out)

        #after convolution
        out = out.view(-1)

        sal_layer2 = sal_out.view(1,227,227)
        sal_out_2 = self.sal_adaptive_pool(sal_layer2)
        sal_out_2 = sal_out_2.permute(1, 2, 0)      

        out = torch.cat((out, sal_out_2.view(-1)))

        out = out.unsqueeze(0)

        out_temp = out
        for i in range(6):
            out_temp = torch.cat((out_temp, out), dim=0)
        out = out_temp

        out = out.unsqueeze(0)
        out = out.permute(1, 0, 2)

        out, _ = self.lstm(out)
        out = out.squeeze(1)

        out = self.linear1(F.relu(out))
        out = self.linear2(F.relu(out))

        return out

    def create_salient_image(self, metadata):
        url = "./POETdataset/PascalImages/boat_" + metadata[0] + ".jpg"
        mbd = psal.get_saliency_mbd(url).astype('uint8')
        mbd = torch.from_numpy(mbd)
        return mbd