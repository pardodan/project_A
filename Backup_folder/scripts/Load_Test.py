import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import cv2
from os import path
import torchvision.models as models

import matplotlib.pyplot as plt
from PIL import Image
import os
import torchvision

################
# Definitions
################
NUM_EPOCHS = 100
Square_Size = 400
Size_x = 600
Size_y = 300
BATCH_SIZE = 64

"""

-- Class TrainQueryDataset: the class that corresponds to create the QueryTrain dataset

"""
#    TO_DO : Try to include this class from "Data.py" #!#!#!#!#!#!#!#!#!#!#!#


class TrainQueryDataset(Dataset):

    def __init__(self, imgs_path, df, our_transforms, suffix='.jpg', preload=False, aug=None, normalization='simple'):

        self.imgs_path = imgs_path
        self.df = df
        self.original_image_names = self.df.id.values
        self.suffix = suffix
        self.aug = aug
        self.normalization = normalization
        self.transforms = our_transforms

        # Run only once:
        if path.exists('query_image_ids.pkl'):
            self.image_ids = torch.load('query_image_ids.pkl')
        else:
            self.image_ids = self.create_image_names()

    #    TO_DO : make this a general func, and use it to make dict for all datasets (- not must...) #!#!#!#!#!#!#!#!#!#
    def create_image_names(self):
        image_ids = {}
        j = 0
        for i in range(self.original_image_names.shape[0]):
            image_id = self.original_image_names[i]
            entry = imgs_path + f'{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}{self.suffix}'
            if path.exists(entry):
                image_ids[j] = image_id
                j += 1

        torch.save(image_ids, 'query_image_ids.pkl')
        return image_ids

    def __getitem__(self, idx):
        id_ = self.image_ids[idx]
        img = self.load_one(id_, self.imgs_path)
        img = img.astype(np.float32)

        if self.normalization:
            img = self.normalize_img(img)

        if self.transforms:
            img = self.transforms(img)

        # feature_dictionary = {'idx': torch.tensor(idx).long(), 'input': img}
        return img

    def __len__(self):
        return len(self.image_ids)

    def load_one(self, id_, imgs_path):
        # try:
        entry = imgs_path + f'{id_[0]}/{id_[1]}/{id_[2]}/{id_}{self.suffix}'
        # print(entry, "1")
        # if path.exists(entry):
        # print(entry, "2")

        src = path.realpath(entry)
        img = cv2.imread(src)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

        # else:
        #     img = np.zeros((Square_Size, Square_Size, 3), dtype=np.int8)
        #     entry = imgs_path + f'{id_[0]}/{id_[1]}/{id_[2]}/{id_}{self.suffix}'
        #     print(entry,"1")
        #     if entry.is_file():
        #         print(entry,"2")
        #
        #         img = cv2.imread(imgs_path + f'{id_[0]}/{id_[1]}/{id_[2]}/{id_}{self.suffix}')
        #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # except:
        #     raise Exception("FAIL READING IMG", imgs_path + f'{id_[0]}/{id_[1]}/{id_[2]}/{id_}{self.suffix}')
        #     print("FAIL READING IMG", imgs_path + f'{id_[0]}/{id_[1]}/{id_[2]}/{id_}{self.suffix}')
        #     img = np.zeros((Square_Size, Square_Size, 3), dtype=np.int8)
        # return img

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
    transforms.ToTensor(),
    transforms.Resize([224, 224])])

"""

-- Create QUERY dataset & encode it into a compressed tensor
-- We perform this stage only once - saves a lot of runtime..!

"""

imgs_path = "/home1/danpardo@staff.technion.ac.il/project/Query_data/"
data = pd.read_csv("/home1/danpardo@staff.technion.ac.il/project/Query_data/train.csv")

TrainQuery_Dataset = TrainQueryDataset(imgs_path, data, normalization='imagenet', our_transforms=data_transforms)
TrainQuery_Dataloader = DataLoader(TrainQuery_Dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)


##########################################################
#                Pretrained ResNet Model                ##
##########################################################

#    TO_DO : Try to use other res-nets to compare different result #!#!#!#!#!#!#!#!#!#!#!#

model = models.resnet18(pretrained=True)
model.fc = Identity()

################
# Check CUDA
################
print("Check if CUDA is available: ", torch.cuda.is_available())

# Define working device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transfer the model to the GPU:
model = model.to(device)

###############
# Run code
###############

# all_tensors = None
all_tensors = torch.zeros((len(TrainQuery_Dataset), 512), device='cpu')

with torch.no_grad():
    for i, batch in enumerate(tqdm(TrainQuery_Dataloader)):
        # Check batch statisticks (should be around mean = 0.0, std = 1.0)
        print(batch.mean(), batch.std())

        # Select first image, and transpose to standard image channels (w, h, c), then normalize to N(0, 1)
        # img = batch[0].numpy().transpose((1, 2, 0))
        # img = (img - img.min()) / (img.max() - img.min())
        # plt.imsave('name.png', img)

        batch = batch.to(device)
        res = model(batch)
        all_tensors[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = res.cpu()

        # if all_tensors is None:
        #     all_tensors = res
        # else:
        #     all_tensors = torch.cat((all_tensors, res), dim=0)


torch.save(all_tensors, 'all_TrainQuery_tensors.pkl')
