import cv2
import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
from imagecorruptions import corrupt
from imagecorruptions import get_corruption_names


## read keypoints and img path as dataframe
def read_data(dirpath_img, dirpath_label=None):
    if dirpath_label:
        keypoints=[]
        for file in os.listdir(dirpath_label):
            varname, ext = file.split('.')
            if ext == 'json':
                with open(dirpath_label+'/'+file) as json_file:
                    data = json.load(json_file)
                    ref_w=data['shapes'][0]['points'][0]        
                    ref_r1=data['shapes'][1]['points'][0] 
                    ref_r2=data['shapes'][2]['points'][0]                       
                    keypoints.append(np.hstack((np.asarray(ref_w),np.asarray(ref_r1),np.asarray(ref_r2))))
        keypoints=np.reshape(np.asarray(keypoints),(-1,6))
    else:
        keypoints=None
    
    # image list
    img_list=[]
    for file in os.listdir(dirpath_img):
        varname, ext = file.split('.')
        if ext == 'png':
            img_list.append(dirpath_img+'/'+file)
                        
    dataset=pd.DataFrame(keypoints) 
    dataset=dataset.rename(columns={0:"ref_w_x", 1:"ref_w_y", 2:"ref_r1_x", 3:"ref_r1_y", 4:"ref_r2_x", 5:"ref_r2_y"})
    dataset['image']=img_list  
    return dataset

## generate heatmap 
IMG_SIZE=256    
def generate_target(joints, image_size=np.array([IMG_SIZE,IMG_SIZE]), heatmap_size=np.array([IMG_SIZE//4,IMG_SIZE//4]),sigma=2):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    num_joints = len(joints)//2
    target_weight = np.ones((num_joints, 1), dtype=np.float32)

    target = np.zeros((num_joints,
                       heatmap_size[1],
                       heatmap_size[0]),
                      dtype=np.float32)
    
    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        feat_stride = image_size / heatmap_size
        mu_x = int(joints[0+joint_id*2] / feat_stride[0] + 0.5)
        mu_y = int(joints[1+joint_id*2] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target

###############################################################################
#============================ augmentation ===================================#
###############################################################################

class Resize(object):
    def __init__(self, img_size=256):
        self.img_size = 256

    def __call__(self, sample):       
        image, keypoints = sample['image'], sample['keypoints']
        image = np.asarray(image,dtype=np.float32)
        if image.shape[0]!=256:
            image = image[416//2-self.img_size//2:416//2+self.img_size//2,416//2-self.img_size//2:416//2+self.img_size//2,:]
            if keypoints is not None:
                keypoints = keypoints-80.
        return {'image': image, 
                'keypoints': keypoints}


class RandomHorizontalFlip(object):
    '''
    Horizontally flip image randomly with given probability
    Args:
        p (float): probability of the image being flipped.
                    Defalut value = 0.5
    '''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):       
        image, keypoints = sample['image'], sample['keypoints']
        if np.random.random() < self.p:
            image = image[:, ::-1]
            if keypoints is not None:
                keypoints[::2] = 256.-keypoints[::2]
        return {'image': image, 
                'keypoints': keypoints}

class Normalize(object):
    '''Normalize input images'''
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        return {'image': image / 255., # scale to [0, 1]
                'keypoints': keypoints}
        
class ToTensor(object):
    '''Convert ndarrays in sample to Tensors.'''
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.reshape(3, IMG_SIZE, IMG_SIZE)
        image = torch.from_numpy(image)
        if keypoints is not None:
            keypoints = generate_target(keypoints)
            keypoints = torch.from_numpy(keypoints)
            return {'image': image, 'keypoints': keypoints}
        else:
            return {'image': image}

class ToTensor_reg(object):
    '''Convert ndarrays in sample to Tensors.'''
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.reshape(3, IMG_SIZE, IMG_SIZE)
        image = torch.from_numpy(image)
        if keypoints is not None:
            keypoints = torch.from_numpy(keypoints)
            return {'image': image, 'keypoints': keypoints}
        else:
            return {'image': image}        
###############################################################################
#============================ visualisation ==================================#
###############################################################################

def show_keypoints(image, keypoints):
    '''
    Show image with keypoints
    Args:
        image (array-like or PIL image): The image data. (M, N)
        keypoints (array-like): The keypoits data. (N, 2)
    '''
      
    plt.imshow(image, cmap='gray')
    if len(keypoints):
        plt.scatter(keypoints[:, 0], keypoints[:, 1], s=24, marker ='.', c='r')
        
def show_images(df, indxs, ncols=5, figsize=(15,10), with_keypoints=True):
   '''
   Show images with keypoints in grids
   Args:
       df (DataFrame): data (M x N)
       idxs (iterators): list, Range, Indexes
       ncols (integer): number of columns (images by rows)
       figsize (float, float): width, height in inches
       with_keypoints (boolean): True if show image with keypoints
   '''
   plt.figure(figsize=figsize)
   nrows = len(indxs) // ncols + 1
   for i, idx in enumerate(indxs):
       image = df.loc[idx, 'image']
       image = Image.open(image)      
       if with_keypoints:
           keypoints = df.loc[idx].drop('image').values.reshape(-1, 2)
       else:
           keypoints = []
       plt.subplot(nrows, ncols, i + 1)
       plt.title(f'Sample #{idx}')
       plt.axis('off')
       plt.tight_layout()
       show_keypoints(image, keypoints)
   plt.show() 
   
###############################################################################
#====================== prepare input data ===================================#
###############################################################################   
class KeypointsDataset(Dataset):
    def __init__(self, dataframe, train=True, transform=None):
        '''
        Args:
            dataframe (DataFrame): data in pandas dataframe format.
            train (Boolean) : True for train data with keypoints, default is True
            transform (callable, optional): Optional transform to be applied on 
            sample
        '''
        self.dataframe = dataframe
        self.train = train
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image = self.dataframe.loc[idx, 'image']
        image = Image.open(image)
        # image = np.asarray(image)
        if self.train:
            keypoints = self.dataframe.iloc[idx, :-1].values.astype(np.float32)
        else:
            keypoints = None
        sample = {'image': image, 'keypoints': keypoints}
        if self.transform:
            sample = self.transform(sample)
        return sample