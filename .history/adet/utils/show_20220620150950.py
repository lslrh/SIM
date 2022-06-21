import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import os
import pdb

def show_feature_map(feature_map, name):
    feature_map = feature_map.squeeze(1)
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    for index in range(1, feature_map_num+1):
        plt.subplot(row_num, row_num, index)
        plt.imshow(feature_map[index-1], cmap='gray')
        plt.axis('off')
        # scipy.misc.imsave(str(index)+".png", feature_map[index-1])
    plt.savefig(os.path.join('feat_visualize', str(name)+".png"))
    plt.show()