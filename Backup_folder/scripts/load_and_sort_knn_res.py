import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

################
# Definitions
################

BEST_QUERIES_NUM = 3
K = 5

"""
1/6 - 
here we took the knn new results, and trying to sort and plot them.
because of loading to here - no need of waiting time.
the plotting here gives us id's of "best" queries with the "best" K-neighbours...
TO DO -- add here also input of "old" knn results, and sort & "plot" them
      -- finish handling the cosine dist
      -- remember to "view" (save) the images by id's, with our new script:
       ./save_selected_images.sh e1f3f5ba42e9a3e0 Query_76
       ./save_selected_images.sh 1841a0457b262a05 Query_76_1st_nn
       and so on...
       (- we have here append that works, but currently with no need - so it is comment..)

"""


# def sort_balltree_results(distances, indices):
#     indices_res = np.zeros((indices.shape[0], indices.shape[1]+1))
#     #dist_res = np.zeros(distances.shape)
#     for query in range(indices.shape[0]):
#         indices_res[query] = np.append(indices[query], np.average(distances[query]))
#
#         #indices[query].append(np.average(distances[query]))
#             # = np.average(distances[query])
#     print(indices_res.shape)
#     print(indices_res)
#
#     indices = np.argsort(indices_res, axis=-1)
#     return indices


def sort_balltree_results(distances, indices):
    print('The input indices:', indices)
    print('The input distances:', distances)

    avg_distances = np.zeros(indices.shape[0])
    sorted_knn_res = np.zeros(indices.shape)
    for query in range(indices.shape[0]):
        avg_distances[query] = np.average(distances[query])

    sorted_idx_to_best_queries = np.argsort(avg_distances, axis=-1)
    # print('avg_distances shape:', avg_distances.shape)
    # print('avg_distances:', avg_distances)
    # print('sorted_idx_to_best_queries:', sorted_idx_to_best_queries)
    for query in range(indices.shape[0]):
        sorted_knn_res[query] = indices[sorted_idx_to_best_queries[query]]
    return sorted_knn_res, sorted_idx_to_best_queries


def sort_cosine_results(dist_mat):
    # print('The input indices:', indices_mat)
    print('The input distances:', dist_mat)

    avg_row_distances = np.zeros(dist_mat.shape[0])         # row here, equivalent to query...
    sorted_idx_to_nn = np.zeros(dist_mat.shape)
    # print(sorted_idx_to_nn.shape)
    for query in range(dist_mat.shape[0]):
        sorted_idx_to_nn[query] = np.argsort(dist_mat[query], axis=-1)
        for nn in range(K):  # maybe we need something similar at the ball-tree func? to do avg only to K...
            # print(sorted_idx_to_nn[query][nn])
            queries_idx_to_nn = int(sorted_idx_to_nn[query][nn])
            # print(dist_mat[query][queries_idx_to_nn])
            # quit()
            avg_row_distances[query] += dist_mat[query][queries_idx_to_nn]
        avg_row_distances[query] /= K

    sorted_idx_to_best_queries = np.argsort(avg_row_distances, axis=-1)

    # sorted_dist_mat = np.zeros((BEST_QUERIES_NUM, K))

    # row_indices = np.arange(cosine_dist_mat.shape[0])
    # row_indices_sorted = np.zeros(cosine_dist_mat.shape)
    # for query in range(indices.shape[0]):
    #     row_indices_sorted[query] = row_indices[sorted_idx_to_best_queries[query]]
    #

    # sorted_knn_res = np.zeros(indices.shape)
    # indices_res = np.zeros(indices.shape)
    # dist_res = np.zeros(distances.shape)
    # for query in range(sorted_dist_mat.shape[0]):
    #     sorted_dist_mat[query] = dist_mat[sorted_idx_to_best_queries[query]]
        # need here locations from the matrix.. how to get?... #indices[sorted_idx_to_best_queries[query]]
    return sorted_idx_to_nn, sorted_idx_to_best_queries


###################################################
# loading old resnet18 results
###################################################
# indices = np.load("/home/danpardo@staff.technion.ac.il/project/indices.npy", mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
# distances = np.load("/home/danpardo@staff.technion.ac.il/project/distances.npy", mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
# cosine_dist_mat = np.load("/home/danpardo@staff.technion.ac.il/project/cosine_mat.npy", mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
# data_csv = pd.read_csv("/home1/danpardo@staff.technion.ac.il/project/index.csv")


###################################################
# loading new resnet101 results
###################################################
indices = np.load("/home/danpardo@staff.technion.ac.il/project/indices_resnet101.npy", mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
distances = np.load("/home/danpardo@staff.technion.ac.il/project/distances_resnet101.npy", mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
cosine_dist_mat = np.load("/home/danpardo@staff.technion.ac.il/project/cosine_mat._resnet101.npy", mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
data_csv = pd.read_csv("/home1/danpardo@staff.technion.ac.il/project/index.csv")

###################################################
# loading new small_index results
###################################################
indices = np.load("/home/danpardo@staff.technion.ac.il/project/indices_index_small_resnet18.npy", mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
distances = np.load("/home/danpardo@staff.technion.ac.il/project/distances_index_small_resnet18.npy", mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
cosine_dist_mat = np.load("/home/danpardo@staff.technion.ac.il/project/cosine_mat.index_small_resnet18.npy", mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
data_csv = pd.read_csv("/home1/danpardo@staff.technion.ac.il/project/index.csv")



#     comparing IDX to IDX sets:     #
# Index_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/all_index_tensors.pkl')
# Query_tensor = torch.load('/home/danpardo@staff.technion.ac.il/project/all_index_tensors.pkl')
# cosine_sim = cosine_similarity(Index_tensor, Query_tensor)
# print('The cosine dist with pytons func:\n', cosine_sim)


###################################################
# inspecting the results of the BALLTREE_KNN method
###################################################

print('The indices matrix with Ball-Tree :\n', indices)
sorted_knn_res, sorted_idx_to_best_queries = sort_balltree_results(distances, indices)

# sorted_idx_to_nn, sorted_idx_to_best_queries = sort_balltree_results(distances, indices)
# row_indices = np.arange(cosine_dist_mat.shape[0])
print('The Ball-Tree Results:\n')
for query in range(BEST_QUERIES_NUM):
    print('The ', K, ' Nearest Neighbours of Query number ', sorted_idx_to_best_queries[query], ' are: ', sorted_knn_res[query,:K])

          # print('The ', K, ' Nearest Neighbours of Query number ', sorted_idx_to_indices[i], ' are: ', indices_sorted[i])
    print('And the relevant ids are:')
    print('Query:', data_csv.at[sorted_idx_to_best_queries[query], 'id'])
    for j in range(K):
        print('Neighbour_', j+1, ':', data_csv.at[sorted_knn_res[query, j], 'id'])

# quit()

###################################################
# inspecting the results of the COSINE_KNN method
###################################################

print('The cosine dist with our func:\n', cosine_dist_mat)
sorted_idx_to_nn, sorted_idx_to_best_queries_cosine = sort_cosine_results(cosine_dist_mat.transpose())
# row_indices = np.arange(cosine_dist_mat.shape[0])
print('The Cosine Results:\n')
for query in range(BEST_QUERIES_NUM):
    print('The ', K, ' Nearest Neighbours of Query number ', sorted_idx_to_best_queries_cosine[query], ' are: ', sorted_idx_to_nn[query][:K])
    print('And the relevant ids are:')
    print('Query:', data_csv.at[sorted_idx_to_best_queries_cosine[query], 'id'])
    for j in range(K):
        queries_idx_to_nn = int(sorted_idx_to_nn[query][j])
        print('Neighbour_', j+1, ':', data_csv.at[queries_idx_to_nn, 'id'])

quit()


