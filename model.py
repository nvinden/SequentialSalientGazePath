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

class HeatMapGenerator(nn.Module):
    def __init__(self, mini_batch_size):
        super(HeatMapGenerator, self).__init__()
        self.mini_batch_size = mini_batch_size

        self.avg_adp_pool = nn.AdaptiveMaxPool2d(227)

        self.image_alex = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
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

        self.salient_alex = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
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

        self.salient_max_pool = nn.AdaptiveMaxPool2d(50)

        self.lin1 = nn.Linear(20932, 10000)
        self.lin2 = nn.Linear(10000, 5184)
    
    def forward(self, image_batch, salient_batch, train=True):
        image_batch.requires_grad_(False)
        salient_batch.requires_grad_(False)

        image = self.avg_adp_pool(image_batch)
        image = self.image_alex(image)
        image = image.view(self.mini_batch_size, -1)

        salient = self.avg_adp_pool(salient_batch)
        salient = self.salient_alex(salient)
        salient = salient.view(self.mini_batch_size, -1)

        salient_raw = self.salient_max_pool(salient_batch)
        salient_raw = salient_raw.view(self.mini_batch_size, -1)

        to_linear = torch.cat((image, salient, salient_raw), dim=1)

        p = 0.3
        out = F.dropout(to_linear, p, training=train)
        out = self.lin1(F.ReLU(out))
        out = F.dropout(out, p, training=train)
        out = self.lin2(F.ReLU(out))
        out = out.view(self.mini_batch_size, 1, 54, 96)

        return out


