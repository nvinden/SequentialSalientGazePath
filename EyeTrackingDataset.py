import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as tf
import pandas as pd
import numpy as np
import math
import scipy.io
from PIL import Image
import cv2
import glob

class EyeTrackingDataset(torch.utils.data.Dataset):
    def __init__(self, root_directory, directory_list, main_directory="Stimuli", phase="Train"):
        self.phase = phase
        trans = tf.ToTensor()

        self.image_list = []
        self.salient_image_list = []
        self.target_list = []

        for directory_name in directory_list:
            filenames_im = glob.glob(root_directory + "/" + main_directory + "/" + directory_name + "/*.jpg")
            filenames_sal = glob.glob(root_directory + "/" + main_directory + "/" + directory_name + "/Output/*.jpg")
            filenames_tar = glob.glob(root_directory + "/" + "FIXATIONMAPS" + "/" + directory_name + "/*.jpg")
            for (filename_im, filename_sal, filename_tar) in zip(filenames_im, filenames_sal, filenames_tar):
                im = Image.open(filename_im).convert("RGB")
                im = trans(im)
                if im.shape == torch.Size([3, 1080, 1920]):
                    self.image_list.append(im)
                else:
                    continue

                im = Image.open(filename_sal)
                im = trans(im)
                if im.shape == torch.Size([1, 141, 250]):
                    self.salient_image_list.append(im)
                else:
                    self.image_list.pop(-1)
                    continue
                
                im = Image.open(filename_tar)
                im = trans(im)
                if im.shape == torch.Size([1, 1080, 1920]):
                    self.target_list.append(im)
                else:
                    self.salient_image_list.pop(-1)
                    self.image_list.pop(-1)
                    continue


            '''
            for filename in glob.glob(root_directory + "/" + main_directory + "/" + directory_name + "/*.jpg"):
                im = Image.open(filename).convert("RGB")
                im = trans(im)
                if im.shape == torch.Size([3, 1080, 1920]):
                    self.image_list.append(im)
                else:
                    continue
            for filename in glob.glob(root_directory + "/" + main_directory + "/" + directory_name + "/Output/*.jpg"):
                im = Image.open(filename)
                im = trans(im)
                self.salient_image_list.append(im)
            for filename in glob.glob(root_directory + "/" + "FIXATIONMAPS" + "/" + directory_name + "/*.jpg"):
                im = Image.open(filename)
                im = trans(im)
                self.target_list.append(im)
            '''

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if self.phase == "Train":
            return {"image": self.image_list[idx], "salient": self.salient_image_list[idx], "target": self.target_list[idx]}
        elif self.phase == "Test":
            return {"image": self.image_list[idx], "salient": self.salient_image_list[idx]}