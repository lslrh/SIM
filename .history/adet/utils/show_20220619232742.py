import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
plt.rcParams['font.sans-serif']=['STSong']
import torchvision.models as models
model = models.alexnet(pretrained=True)
import pdb

#2. 获取第k层的特征图
'''
args:
k:定义提取第几层的feature map
x:图片的tensor
model_layer：是一个Sequential()特征层
'''
def get_k_layer_feature_map(model_layer, k, x):
    with torch.no_grad():
        for index, layer in enumerate(model_layer):#model的第一个Sequential()是有多层，所以遍历
            x = layer(x)#torch.Size([1, 64, 55, 55])生成了64个通道
            if k == index:
                return x


#  可视化特征图
def show_feature_map(feature_map):#feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
                                                                         # feature_map[2].shape     out of bounds
    # feature_map = feature_map.squeeze(0)#压缩成torch.Size([64, 55, 55])
    
    #以下4行，通过双线性插值的方式改变保存图像的大小

    feature_map =feature_map.view(1,feature_map.shape[0],feature_map.shape[1],feature_map.shape[2])#(1,64,55,55)
    upsample = torch.nn.UpsamplingBilinear2d(size=(256,256))#这里进行调整大小
    feature_map = upsample(feature_map)
    feature_map = feature_map.view(feature_map.shape[1],feature_map.shape[2],feature_map.shape[3])
    
    feature_map_num = feature_map.shape[0]#返回通道数
    row_num = np.ceil(np.sqrt(feature_map_num))#8
    plt.figure()
    for index in range(1, feature_map_num + 1):#通过遍历的方式，将64个通道的tensor拿出

        plt.subplot(row_num, row_num, index)
        # plt.imshow(feature_map[index - 1], cmap='gray')#feature_map[0].shape=torch.Size([55, 55])
        #将上行代码替换成，可显示彩色 plt.imshow(transforms.ToPILImage()(feature_map[index - 1]))#feature_map[0].shape=torch.Size([55, 55])
        plt.axis('off')
    plt.savefig( 'feature_map_save//'+str(index) + ".png")
    plt.show()
