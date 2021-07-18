from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import cv2
import torchvision.models as models
import os
import csv
from os import path
from PIL import Image, ImageOps
import pandas as pd

################
# Definitions
################
NUM_EPOCHS = 100
Square_Size = 400
Size_x = 600
Size_y = 300
BATCH_SIZE = 8
TensorSize = 512   # for resnet 18
#TensorSize = 2048   # for resnet 101

####################################################################################################
#                                  Create Clean Dataloader                                         #
####################################################################################################

class Small_Dataset(Dataset):
    def __init__(self, imgs_path, DF_test_dir, our_transforms, normalization='simple', suffix='.jpg'):

        self.imgs_path = imgs_path
        self.df = DF_test_dir
        self.image_names = self.df.id.values
        self.suffix = suffix
        self.normalization = normalization
        self.transforms = our_transforms

    def __getitem__(self, idx):
        id_ = self.image_names[idx]
        imgs_path_ = self.imgs_path
        img = self.load_one(id_, imgs_path_)

        img = img.astype(np.float32)

        if self.normalization:
            img = self.normalize_img(img)

        if self.transforms:
            # type(self.transforms)
            # img = resize_with_padding(img, (480, 480))
            img = self.transforms(img)

        return img, torch.tensor(idx)

    def __len__(self):
        return len(self.image_names)

    def load_one(self, id_, imgs_path):

        img = cv2.imread(imgs_path + f'{id_[0]}/{id_[1]}/{id_[2]}/{id_}{self.suffix}')
        img = cv2.resize(img, (480, 480), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

        # def __create_dictionary__(self):
        #     for (i all images...len.. )
        #         img[i] = self.__getitem__()
        #         tensor = self.to_torch_tensor(img)
        #         feature_dictionary = {'idx': torch.tensor(idx).long(),
        #                     'input': tensor}
        #     return feature_dictionary

        # def augment(self, img):
        #     img_aug = self.aug(image=img)['image']
        #     return img_aug.astype(np.float32)

    def normalize_img(self, img):

        if self.normalization == 'channel':
            pixel_mean = img.mean((0, 1))
            pixel_std = img.std((0, 1)) + self.eps
            img = (img - pixel_mean[None, None, :]) / pixel_std[None, None, :]
            img = img.clip(-20, 20)

        elif self.normalization == 'channel_mean':
            pixel_mean = img.mean((0, 1))
            img = (img - pixel_mean[None, None, :])
            img = img.clip(-20, 20)

        elif self.normalization == 'image':
            img = (img - img.mean()) / img.std() + self.eps
            img = img.clip(-20, 20)

        elif self.normalization == 'simple':
            img = img / 255

        elif self.normalization == 'inception':

            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            img = img.astype(np.float32)
            img = img / 255.
            img -= mean
            img *= np.reciprocal(std, dtype=np.float32)

        elif self.normalization == 'imagenet':

            mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            std = np.array([58.395, 57.120, 57.375], dtype=np.float32)
            img = img.astype(np.float32)
            img -= mean
            img *= np.reciprocal(std, dtype=np.float32)

        else:
            pass

        return img

    def to_torch_tensor(self, img):
        return torch.from_numpy(img.transpose((2, 0, 1)))


"""

-- Class Identity: uses for extracting the last layer from resnet model

"""


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


"""

-- data_transforms: all transformations when uploading the data

"""
data_transforms = transforms.Compose([
    transforms.ToTensor()])
    # transforms.Resize([224, 224])])


"""

-- Create CleanSmall dataset & encode it into a compressed tensor
-- We perform this stage only once - saves a lot of runtime..!

"""

imgs_path = "/home1/danpardo@staff.technion.ac.il/project/"
DF_test_dir = pd.read_csv("/home1/danpardo@staff.technion.ac.il/project/index_clean_small.csv")

index_small_dataset = Small_Dataset(imgs_path, DF_test_dir, our_transforms=data_transforms, normalization='imagenet')
index_small_dataloader = DataLoader(index_small_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

###############################
#   Pretrained ResNet model   #
###############################

#    TO_DO : Try to use other res-nets to compare different result #!#!#!#!#!#!#!#!#!#!#!#
# model = models.resnet101(pretrained=True)
model = models.resnet18(pretrained=True)
model.fc = Identity()

##################
#   Check CUDA   #
##################
print("Check if CUDA is available: ", torch.cuda.is_available())

# Define working device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transfer the model to the GPU:
model = model.to(device)

################
#   Run code   #
################

all_tensors = torch.zeros((len(index_small_dataset), TensorSize), device='cpu')

with torch.no_grad():
    for i, (batch, idx) in enumerate(tqdm(index_small_dataloader)):

        batch = batch.to(device)
        res = model(batch)
        all_tensors[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = res.cpu()

torch.save(all_tensors, 'index_small_clean_tensors_resnet18.pkl')


