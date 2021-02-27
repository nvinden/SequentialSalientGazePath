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


def draw_picture_points(image, metadata, eye_points, filename):
    img = np.reshape(image, (metadata[1], metadata[2], -1))
    img = np.float32(img)
    img = cv2.circle(img, tuple(eye_points[0][0:2]), 3, (255,0,0), 10)
    img = cv2.circle(img, tuple(eye_points[0][2:4]), 3, (255,0,0), 10)
    img = cv2.circle(img, tuple(eye_points[1][0:2]), 3, (255,128,0), 10)
    img = cv2.circle(img, tuple(eye_points[1][2:4]), 3, (255,128,0), 10)
    img = cv2.circle(img, tuple(eye_points[2][0:2]), 3, (255,255,0), 10)
    img = cv2.circle(img, tuple(eye_points[2][2:4]), 3, (255,255,0), 10)
    img = cv2.circle(img, tuple(eye_points[3][0:2]), 3, (128,255,0), 10)
    img = cv2.circle(img, tuple(eye_points[3][2:4]), 3, (128,255,0), 10)
    img = cv2.circle(img, tuple(eye_points[4][0:2]), 3, (0,0,255), 10)
    img = cv2.circle(img, tuple(eye_points[4][2:4]), 3, (0,0,255), 10)
    img = cv2.circle(img, tuple(eye_points[5][0:2]), 3, (204,0,204), 10)
    img = cv2.circle(img, tuple(eye_points[5][2:4]), 3, (204,0,204), 10)
    img = cv2.circle(img, tuple(eye_points[6][0:2]), 3, (255,255,255), 10)
    img = cv2.circle(img, tuple(eye_points[6][2:4]), 3, (255,255,255), 10)
    img = np.flip(img, 2)
    
    cv2.imwrite("/home/nvinden/ML/EyeTracking/Images/" + filename + ".jpg", img)

