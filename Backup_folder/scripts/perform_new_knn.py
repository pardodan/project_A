import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np


################
# Definitions
################






NUM_QUERIES_FOR_COSINE_KNN = 500
NUM_QUERIES_FOR_BALLTREE_KNN = 500
K = 5

"""
1/6 - 
we tried here faster methods for performing knn.
it worked, preety fast, but not so good results - to look at the res after the following code, at "load&sort knn res"
here, we added saving & loading mp arrays with the results
to remember - the debugger is working..
            - we have in comment, several options of sets to k-nn search between...
TO DO -- try to better handle the cosine and see its results
      -- to load a cleaner idx or train dataset - then try run it again
      -- try encode with some other resnet (before getting to this code..)
      -- try to play and increase the "hyper-params" here (and maybe change their names)
"""


def cosine(mat_1, mat_2):
    dist = mat_1 @ mat_2.transpose()
    norm_1 = np.expand_dims((mat_1 * mat_1).sum(1), axis=1)
    norm_2 = np.expand_dims((mat_2 * mat_2).sum(1), axis=1)

    norm_1 = np.repeat(norm_1, mat_2.shape[0], axis=1)
    norm_2 = np.repeat(norm_2, mat_1.shape[0], axis=1).transpose()

    dist = dist / (norm_1 * norm_2)
    return dist


# #     comparing TRAIN (as query) to IDX sets:     #
# Index_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/all_index_tensors.pkl')
# Query_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/all_Query_tensors.pkl')


# #     comparing TRAIN to TRAIN sets:     #
# Index_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/all_Query_tensors.pkl')
# Query_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/all_Query_tensors.pkl')
# query_image_ids = torch.load('query_image_ids.pkl')


# #     comparing IDX to IDX sets:     #
# Index_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/all_index_tensors.pkl')
# Query_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/all_index_tensors.pkl')
# data_csv = pd.read_csv("/home1/danpardo@staff.technion.ac.il/project/index.csv")

#
# #     comparing IDX to IDX sets, with no resize:     #
# Index_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/all_index_tensors_noresize.pkl')
# Query_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/all_index_tensors_noresize.pkl')
# data_csv = pd.read_csv("/home1/danpardo@staff.technion.ac.il/project/index.csv")


#     comparing IDX to IDX sets, with no resize & new resnet101:     #
Index_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/all_index_tensors_noresize_other_resnet.pkl')
Query_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/all_index_tensors_noresize_other_resnet.pkl')
#data_csv = pd.read_csv("/home1/danpardo@staff.technion.ac.il/project/index.csv")


#     comparing IDX to IDX sets, with no resize & new resnet101:     #
#Index_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/index_small_clean_tensors_resnet18.pkl')
#Query_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/index_small_clean_tensors_resnet18.pkl')
#data_csv = pd.read_csv("/home1/danpardo@staff.technion.ac.il/project/index.csv")


#     converting to numpy:     #
Index_tensor = Index_tensor.numpy()
Query_tensor = Query_tensor.numpy()

#     performing the k-nn     #
cosine_dist = cosine(Index_tensor, Query_tensor[:NUM_QUERIES_FOR_COSINE_KNN])

Neighbors = NearestNeighbors(n_neighbors=K, algorithm='ball_tree', n_jobs=8).fit(Index_tensor)
distances, indices = Neighbors.kneighbors(Query_tensor[:NUM_QUERIES_FOR_BALLTREE_KNN])

#     saving the results before sorting them:     #
np.save("indices_resnet101.npy", indices, allow_pickle=True, fix_imports=True)
np.save("distances_resnet101.npy", distances, allow_pickle=True, fix_imports=True)
np.save("cosine_mat._resnet101.npy", cosine_dist, allow_pickle=True, fix_imports=True)

