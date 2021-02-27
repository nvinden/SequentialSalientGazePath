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

desired_length = 7

out = np.ndarray((md_df.shape[0], 7, 4), dtype=float)

for row_num, row in enumerate(md_df):
    best = 100
    best_idx = 0
    for i in range(5):
        points = row[3+2*i]
        length = points.shape[0]
        if abs(length - desired_length) < abs(best):
            best = length - desired_length
            best_idx = 3+2*i

    to_input = np.concatenate((row[best_idx], row[best_idx + 1]), 1)
    
    if best == 0:
        out[row_num] = to_input
    elif best > 0:
        out[row_num] = to_input[0:7]
    elif best < 0:
        max = 0
        data = to_input
        for i in range(7):
            if i < data.shape[0]:
                out[row_num][i] = data[i]
                max = i
            else:
                out[row_num][i] = data[max]

    width = row[2]
    height = row[1]
    for i in range(desired_length):
        out[row_num][i][0] = 227 * out[row_num][i][0] / width
        out[row_num][i][2] = 227 * out[row_num][i][2] / width
        out[row_num][i][1] = 227 * out[row_num][i][1] / height
        out[row_num][i][3] = 227 * out[row_num][i][3] / height


with open('best_table.npy', 'wb') as f:
    np.save(f, out)
    