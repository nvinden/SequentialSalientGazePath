import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
from scipy.io import loadmat
from PIL import Image

#parameters
directory = "./POETdataset/PascalImages/"
prefix = "boat_"
image_type = "jpg"

with open('metadata_table.npy', 'rb') as f:
    df = np.load(f, allow_pickle=True)

print(df.shape)

out_array = np.empty((df.shape[0]), dtype=object)

for i in range(504):
    row = df[i]
    img_path = directory + prefix + row[0] + "." + image_type
    an_image = Image.open(img_path)
    image_sequence = an_image.getdata()
    image_array = np.array(image_sequence)
    out_array = np.append(out_array, image_array)
    out_array[i] = image_array

print(out_array[0].shape)

with open('picture_table.npy', 'wb') as f:
    np.save(f, out_array)
    
    