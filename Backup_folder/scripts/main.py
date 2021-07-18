import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
NUM_EPOCHS = 100
Square_Size = 400
Size_x = 600
Size_y = 300

##########################################################
#                         MAIN                          ##
##########################################################


###############################
#    check specific images    #
###############################
imgs_path = "/home1/danpardo@staff.technion.ac.il/project/Query_data/"
ids = torch.load('/home/danpardo@staff.technion.ac.il/project/query_image_ids.pkl')
suffix = '.jpg'
INDEX_csv = pd.read_csv("/home1/danpardo@staff.technion.ac.il/project/index.csv")

print("the 1st query and neighbours")
print(ids[1])

print(INDEX_csv.id[502716])
print(INDEX_csv.id[481933])
print(INDEX_csv.id[278868])

print("the 2nd query and neighbours")
print(ids[2])
print(INDEX_csv.id[554185])
print(INDEX_csv.id[738988])
print(INDEX_csv.id[436606])

print("the 3rd query and neighbours")
print(ids[3])
print(INDEX_csv.id[101996])
print(INDEX_csv.id[746211])
print(INDEX_csv.id[233366])

#
# entry = imgs_path + f'{ids[0][0]}/{ids[0][1]}/{ids[0][2]}/{ids[0]}{suffix}'
# print(entry)
# import scipy.io as sio
#
# im1 = cv2.imread(entry)
# # copy_img = np.copy(im)
# plt.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
# _ = plt.axis('off')
