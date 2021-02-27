import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
from scipy.io import loadmat
from PIL import Image
import pyimgsaliency.pyimgsaliency as psal
import cv2

with open('metadata_table.npy', 'rb') as f:
    md_df = np.load(f, allow_pickle=True)

with open('picture_table.npy', 'rb') as f:
    pic_df = np.load(f, allow_pickle=True)

out_arr = np.empty((md_df.shape[0], 10), dtype=object)

for i in range(md_df.shape[0]):
    print(i+1)
    md_row = md_df[i]
    pic_row = pic_df[i]
    # get the saliency maps using the 3 implemented methods
    url = "./POETdataset/PascalImages/boat_" + md_row[0] + ".jpg"
    try:
        mbd = psal.get_saliency_mbd(url).astype('uint8')
    except:
        print("Error: MBD")
        continue

    for j in range(3,13):
        curr_pts = np.empty((md_row[j].shape[0]), dtype=int)
        for k in range(md_row[j].shape[0]):
            x = round(md_row[j][k][0])
            y = round(md_row[j][k][1])
            x_min = 0
            x_max = mbd.shape[1]
            y_min = 0
            y_max = mbd.shape[0]
            if x > x_max:
                x = x_max
            if x < x_min:
                x = x_min
            if y > y_max:
                y = y_max
            if y < y_min:
                y = y_min
            curr_pts[k] = mbd[y - 1][x - 1]
        out_arr[i][j - 3] = curr_pts

with open('salient_table.npy', 'wb') as f:
    np.save(f, out_arr)

#NEED TO FIX THE POINTS
            
            
    
    

