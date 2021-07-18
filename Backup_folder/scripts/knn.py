import torch
import numpy as np

from matplotlib.colors import ListedColormap
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2
from PIL import Image
import torchvision.models as models

################
# Definitions
################
K = 4


"""
19/5 try - with this dist:
we got no good results - to comp in the folder, "gad.jpg" "dan.jpg" and similars...
"""
def euclidean_distance(x1, x2):
    # print(x1.size(0))
    # print(x2.size(0))
    # return (x1 - x2).t @ (x1 - x2)  # this one for matrices - not tensors..
    return ((x1 - x2)**2).sum()


"""
afterwards - we changed the dist and let it run. to check the new results...! - didnt do it yet...
and maybe the problem will be solved if we will take real query images (not train...) because maybe not much matches...
or maybe first - try to implement the knn only on images from train - then see if it caught images from same class...
"""


class KNN:
    def __init__(self, k, Index_):
        self.k = k
        self.Index = Index_

        # Saves distances between queries and indices
        # self.distances = np.zeros(self.Index.size(0), dtype=np.float32)

    def predict(self, Query_img):

        distances = np.zeros(self.Index.size(0), dtype=np.float32)
        # print('amount of idices to check:', self.Index.size(0))
        for i in range(self.Index.size(0)):
            # print('the current index img is:', i)
            distances[i] = euclidean_distance(Query_img, self.Index[i])
        sorted_distances = np.argsort(distances)
        # print(sorted_distances[self.k+1:])
        # print(sorted_distances[self.k:])
        # print(sorted_distances[:self.k+1])
        return sorted_distances[:self.k]

# def accuracy(y_true, y_pred):
#     accuracy = np.sum(y_true == y_pred) / len(y_true)
#     return accuracy


# cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

##########################################################
## ---------------------  MAIN ------------------------ ##
##########################################################
#
# # ------  check - taking train set as the index set ------#
# Index_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/all_TrainQuery_tensors.pkl')
#
# #
# # # to load : /home/danpardo@staff.technion.ac.il/project/all_index_tensors.pkl
# # Index_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/all_index_tensors.pkl')
#
# # to load : /home/danpardo@staff.technion.ac.il/project/all_TrainQuery_tensors.pkl
# TrainQuery_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/all_TrainQuery_tensors.pkl') ## is this the path??
#



# #     comparing TRAIN (as query) to IDX sets:     #
# Index_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/all_index_tensors.pkl')
# TrainQuery_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/all_TrainQuery_tensors.pkl') ## is this the path??
#
# #     comparing TRAIN to TRAIN sets:     #
# Index_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/all_TrainQuery_tensors.pkl')
# TrainQuery_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/all_TrainQuery_tensors.pkl') ## is this the path??
# query_image_ids = torch.load('query_image_ids.pkl')



#     comparing IDX to IDX sets:     #
Index_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/all_index_tensors.pkl')
TrainQuery_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/all_index_tensors.pkl') ## is this the path??
data = pd.read_csv("/home1/danpardo@staff.technion.ac.il/project/index.csv")



knn_inst = KNN(K, Index_tensor)
Predictions = np.zeros((TrainQuery_tensor.size(0), K))
# for i, batch in enumerate(tqdm(TrainQuery_Dataloader)):





# Predictions[0] = knn_inst.predict(TrainQuery_tensor[0])
# print(Predictions[0])






#
#
for i in range(TrainQuery_tensor.size(0)):
    print(i)
    Predictions[i] = knn_inst.predict(TrainQuery_tensor[i])
    print(Predictions[i])

print(Predictions)
torch.save(Predictions, 'predictions_dist_1st_knn_tensors.pkl')

# query_image_ids = torch.load('query_image_ids.pkl')

# sample = 0
# predicted_images_ids = Predictions[0]
#










#    TO_DO : show several queries + their neighbours #!#!#!#!#!#!#!#!#!#!#!#
# limit = [1, 2, 3]
# suffix = '.jpg'
# # allocate??
# Query = np.zeros(())
# Neighbours = [][]
# for i in limit:
#     Query[i] = knn_inst.TrainQuery_tensor[i].numpy().transpose((1, 2, 0))
#     Query[i] = (Query[i] - Query[i].min()) / (Query[i].max() - Query[i].min())
#     plt.imsave('Query' + f'{i}{suffix}', Query[i])
#     for j in limit:
#         Neighbours[i][j] = TrainQuery_tensor[Predictions[i][j]].numpy().transpose((1, 2, 0))
#         Neighbours[i][j] = (Neighbours[i][j] - Neighbours[i][j].min()) / (Neighbours[i][j].max() - Neighbours[i][j].min())
#         plt.imsave('Neighbours' + f'{i}{j}{suffix}', Neighbours[i][j])


