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

with open('metadata_table.npy', 'rb') as f:
    md_df = np.load(f, allow_pickle=True)

with open('picture_table_debug.npy', 'rb') as f:
    pic_df = np.load(f, allow_pickle=True)

with open('salient_table_debug.npy', "rb") as f:
    sal_df = np.load(f, allow_pickle=True)

with open('best_table.npy', "rb") as f:
    best_df = np.load(f, allow_pickle=True)

model = SalientEyePointGenerator()
model.load_state_dict(torch.load("/home/nvinden/ML/EyeTracking/test.model"))

for i in range(5):
    print(best_df[i])
    print(model(md_df[i], pic_df[i], sal_df[i]))