import numpy as np
import collections
import sys
import random
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from models.LightPointNet import LightPointNet
from models.PoseResNet import BasicBlock, Bottleneck, PoseResNet
from models.BlazePose import BlazePose
from models.ReceptionNet import ReceptionNet
from utils.utils_inference import predict, get_max_preds, get_joint_location_result, save_output
from utils.dataset import read_data, KeypointsDataset, Normalize, ToTensor_reg, ToTensor, Resize
from utils import hparams_registry, misc
from utils.evaluation import evaluation

###############################################################################
#============================ comparing DL methods ===========================#
###############################################################################
# arguments can be given in comment line by argparse or in a dictionary like below
args_dict = {'data_test': 'test256_aug', # 'test256','test256_aug','test256_cornercases', 'test256_new'
             'image_size': 256,
             'output_type': 'coordinates', # 'heatmap','coordinates', 'coordinates_direct'
             'netwrok': 'LightPointNet_normal', # 'PoseResNet', 'BlazePose', 'ReceptionNet', 'LightPointNet_normal', 'LightPointNet_large', 'LightPointNet_small'
             'model_path': './trained_models/LightPointNet_normal_Integral_L1.pt', #
             'hparams': None
            }
LightPointNet_normal_Pw,  LightPointNet_normal_Pr1, LightPointNet_normal_Pr2 = evaluation(args_dict)
    
args_dict = {'data_test': 'test256_aug', # 'test256','test256_aug','test256_cornercases', 'test256_new'
             'image_size': 256,
             'output_type': 'coordinates', # 'heatmap','coordinates', 'coordinates_direct'
             'netwrok': 'ReceptionNet', # 'PoseResNet', 'BlazePose', 'ReceptionNet', 'LightPointNet_normal', 'LightPointNet_large', 'LightPointNet_small'
             'model_path': './trained_models/ReceptionNet_Integral_L1.pt', #
             'hparams': None
            }
ReceptionNet_Pw,  ReceptionNet_Pr1, ReceptionNet_Pr2 = evaluation(args_dict)

args_dict = {'data_test': 'test256_aug', # 'test256','test256_aug','test256_cornercases', 'test256_new'
             'image_size': 256,
             'output_type': 'coordinates_direct', # 'heatmap','coordinates', 'coordinates_direct'
             'netwrok': 'BlazePose', # 'PoseResNet', 'BlazePose', 'ReceptionNet', 'LightPointNet_normal', 'LightPointNet_large', 'LightPointNet_small'
             'model_path': './trained_models/BlazePose_stage1.pt', #
             'hparams': None
            }
BlazePose_Pw,  BlazePose_Pr1, BlazePose_Pr2 = evaluation(args_dict)

args_dict = {'data_test': 'test256_aug', # 'test256','test256_aug','test256_cornercases', 'test256_new'
             'image_size': 256,
             'output_type': 'heatmap', # 'heatmap','coordinates', 'coordinates_direct'
             'netwrok': 'PoseResNet', # 'PoseResNet', 'BlazePose', 'ReceptionNet', 'LightPointNet_normal', 'LightPointNet_large', 'LightPointNet_small'
             'model_path': './trained_models/PoseResNet_MSE.pt', #
             'hparams': None
            }
PoseResNet_Pw,  PoseResNet_Pr1, PoseResNet_Pr2 = evaluation(args_dict)

gray   = '#A5A5A5'
red    = '#ED7D31'
yellow = '#FBBE00'
green  = '#70AD47'
blue   = '#75ABDC'
black  = '#000000'
num = LightPointNet_normal_Pw[0] + LightPointNet_normal_Pw[1] + LightPointNet_normal_Pw[2] + LightPointNet_normal_Pw[3]
histo={'0-1 pixel':[LightPointNet_normal_Pw[0], PoseResNet_Pw[0], ReceptionNet_Pw[0], BlazePose_Pw[0]],'1-5 pixels':[LightPointNet_normal_Pw[1], PoseResNet_Pw[1], ReceptionNet_Pw[1], BlazePose_Pw[1]],'5-20 pixels':[LightPointNet_normal_Pw[2], PoseResNet_Pw[2], ReceptionNet_Pw[2], BlazePose_Pw[2]],'miss detection':[LightPointNet_normal_Pw[3], PoseResNet_Pw[3], ReceptionNet_Pw[3], BlazePose_Pw[3]]}
df=pd.DataFrame.from_dict(histo, orient='index',columns=['LightPointNet','PoseResNet','ReceptionNet','BlazePose'])
fig=df.plot(kind='bar',color=[red,blue,yellow,green],figsize=(12, 6))
for p in fig.patches:
    value = p.get_height()/num
    fig.annotate("{:.1%}".format(value), (p.get_x() * 1.005, p.get_height() * 1.005), fontsize = 8)
plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', fontsize='large', ncol=4, handletextpad=0.3, borderpad=0.2)                          
plt.xticks(rotation=0)
plt.ylabel('Nums of images')
plt.tight_layout()
# fig.figure.savefig('./figures/error_Pw_images.png',dpi=600)

histo={'0-1 pixel':[LightPointNet_normal_Pr1[0], PoseResNet_Pr1[0], ReceptionNet_Pr1[0], BlazePose_Pr1[0]],'1-5 pixels':[LightPointNet_normal_Pr1[1], PoseResNet_Pr1[1], ReceptionNet_Pr1[1], BlazePose_Pr1[1]],'5-20 pixels':[LightPointNet_normal_Pr1[2], PoseResNet_Pr1[2], ReceptionNet_Pr1[2], BlazePose_Pr1[2]],'miss detection':[LightPointNet_normal_Pr1[3], PoseResNet_Pr1[3], ReceptionNet_Pr1[3], BlazePose_Pr1[3]]}
df=pd.DataFrame.from_dict(histo, orient='index',columns=['LightPointNet','PoseResNet','ReceptionNet','BlazePose'])
fig=df.plot(kind='bar',color=[red,blue,yellow,green],figsize=(12, 6))
for p in fig.patches:
    value = p.get_height()/num
    fig.annotate("{:.1%}".format(value), (p.get_x() * 1.005, p.get_height() * 1.005), fontsize = 8)
plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', fontsize='large', ncol=4, handletextpad=0.3, borderpad=0.2)                          
plt.xticks(rotation=0)
plt.ylabel('Nums of images')
plt.tight_layout()
# fig.figure.savefig('./figures/error_Pr1_images.png',dpi=600)

histo={'0-1 pixel':[LightPointNet_normal_Pr2[0], PoseResNet_Pr2[0], ReceptionNet_Pr2[0], BlazePose_Pr2[0]],'1-5 pixels':[LightPointNet_normal_Pr2[1], PoseResNet_Pr2[1], ReceptionNet_Pr2[1], BlazePose_Pr2[1]],'5-20 pixels':[LightPointNet_normal_Pr2[2], PoseResNet_Pr2[2], ReceptionNet_Pr2[2], BlazePose_Pr2[2]],'miss detection':[LightPointNet_normal_Pr2[3], PoseResNet_Pr2[3], ReceptionNet_Pr2[3], BlazePose_Pr2[3]]}
df=pd.DataFrame.from_dict(histo, orient='index',columns=['LightPointNet','PoseResNet','ReceptionNet','BlazePose'])
fig=df.plot(kind='bar',color=[red,blue,yellow,green],figsize=(12, 6))
for p in fig.patches:
    value = p.get_height()/num
    fig.annotate("{:.1%}".format(value), (p.get_x() * 1.005, p.get_height() * 1.005), fontsize = 8)
plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', fontsize='large', ncol=4, handletextpad=0.3, borderpad=0.2)                          
plt.xticks(rotation=0)
plt.ylabel('Nums of images')
plt.tight_layout()
# fig.figure.savefig('./figures/error_Pr2_images.png',dpi=600)

## averaged
LightPointNet_normal_avg = (LightPointNet_normal_Pw + LightPointNet_normal_Pr1 + LightPointNet_normal_Pr2)/3
PoseResNet_avg = (PoseResNet_Pw + PoseResNet_Pr1 + PoseResNet_Pr2)/3
ReceptionNet_avg = (ReceptionNet_Pw + ReceptionNet_Pr1 + ReceptionNet_Pr2)/3
BlazePose_avg = (BlazePose_Pw + BlazePose_Pr1 + BlazePose_Pr2)/3

histo={'0-1 pixel':[LightPointNet_normal_avg[0], PoseResNet_avg[0], ReceptionNet_avg[0], BlazePose_avg[0]],'1-5 pixels':[LightPointNet_normal_avg[1], PoseResNet_avg[1], ReceptionNet_avg[1], BlazePose_avg[1]],'5-20 pixels':[LightPointNet_normal_avg[2], PoseResNet_avg[2], ReceptionNet_avg[2], BlazePose_avg[2]],'miss detection':[LightPointNet_normal_avg[3], PoseResNet_avg[3], ReceptionNet_avg[3], BlazePose_avg[3]]}
df=pd.DataFrame.from_dict(histo, orient='index',columns=['LightPointNet','PoseResNet','ReceptionNet','BlazePose'])
fig=df.plot(kind='bar',color=[red,blue,yellow,green],figsize=(12, 6))
for p in fig.patches:
    value = p.get_height()/num
    fig.annotate("{:.1%}".format(value), (p.get_x() * 1.005, p.get_height() * 1.005), fontsize = 8)
plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', fontsize='large', ncol=4, handletextpad=0.3, borderpad=0.2)                          
plt.xticks(rotation=0)
plt.ylabel('Nums of images')
plt.tight_layout()
fig.figure.savefig('./figures/error_avg_images.png',dpi=600)


###############################################################################
#========================== comparing scaled methods =========================#
###############################################################################
# args_dict = {'data_test': 'test256', # 'test256','test256_aug','test256_cornercases', 'test256_new'
#              'image_size': 256,
#              'output_type': 'coordinates', # 'heatmap','coordinates', 'coordinates_direct'
#              'netwrok': 'LightPointNet_normal', # 'PoseResNet', 'BlazePose', 'ReceptionNet', 'LightPointNet_normal', 'LightPointNet_large', 'LightPointNet_small'
#              'model_path': './trained_models/LightPointNet_normal_Integral_L1.pt', #
#              'hparams': None
#             }
# LightPointNet_normal_Pw,  LightPointNet_normal_Pr1, LightPointNet_normal_Pr2 = evaluation(args_dict)

# args_dict = {'data_test': 'test256', # 'test256','test256_aug','test256_cornercases', 'test256_new'
#              'image_size': 256,
#              'output_type': 'coordinates', # 'heatmap','coordinates', 'coordinates_direct'
#              'netwrok': 'LightPointNet_large', # 'PoseResNet', 'BlazePose', 'ReceptionNet', 'LightPointNet_normal', 'LightPointNet_large', 'LightPointNet_small'
#              'model_path': './trained_models/LightPointNet_large_Integral_L1.pt', #
#              'hparams': None
#             }
# LightPointNet_large_Pw,  LightPointNet_large_Pr1, LightPointNet_large_Pr2 = evaluation(args_dict)

# args_dict = {'data_test': 'test256', # 'test256','test256_aug','test256_cornercases', 'test256_new'
#              'image_size': 256,
#              'output_type': 'coordinates', # 'heatmap','coordinates', 'coordinates_direct'
#              'netwrok': 'LightPointNet_small', # 'PoseResNet', 'BlazePose', 'ReceptionNet', 'LightPointNet_normal', 'LightPointNet_large', 'LightPointNet_small'
#              'model_path': './trained_models/LightPointNet_small_Integral_L1.pt', #
#              'hparams': None
#             }
# LightPointNet_small_Pw,  LightPointNet_small_Pr1, LightPointNet_small_Pr2 = evaluation(args_dict)

# LightPointNet_normal_avg = (LightPointNet_normal_Pw + LightPointNet_normal_Pr1 + LightPointNet_normal_Pr2)/3
# LightPointNet_small_avg = (LightPointNet_small_Pw + LightPointNet_small_Pr1 + LightPointNet_small_Pr2)/3
# LightPointNet_large_avg = (LightPointNet_large_Pw + LightPointNet_large_Pr1 + LightPointNet_large_Pr2)/3

# histo={'0-1 pixel':[LightPointNet_normal_avg[0], LightPointNet_small_avg[0], LightPointNet_large_avg[0]],'1-5 pixels':[LightPointNet_normal_avg[1], LightPointNet_small_avg[1], LightPointNet_large_avg[1]],'5-20 pixels':[LightPointNet_normal_avg[2], LightPointNet_small_avg[2], LightPointNet_large_avg[2]],'miss detection':[LightPointNet_normal_avg[3], LightPointNet_small_avg[3], LightPointNet_large_avg[3]]}
# df=pd.DataFrame.from_dict(histo, orient='index',columns=['LightPointNet','LightPointNet_small','LightPointNet_large'])
# fig=df.plot(kind='bar',color=[red,blue,yellow],figsize=(12, 6))
# for p in fig.patches:
#     value = p.get_height()/num
#     fig.annotate("{:.1%}".format(value), (p.get_x() * 1.005, p.get_height() * 1.005), fontsize = 8)
# plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', fontsize='large', ncol=3, handletextpad=0.3, borderpad=0.2)                          
# plt.xticks(rotation=0)
# plt.ylabel('Nums of images')
# plt.tight_layout()

# fig.figure.savefig('./figures/error_avg_images.png',dpi=600)

# ###############################################################################
# #============================== Complexity ===================================#
# ###############################################################################
# from ptflops import get_model_complexity_info

# # PoseResnet
# resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
#                 34: (BasicBlock, [3, 4, 6, 3]),
#                 50: (Bottleneck, [3, 4, 6, 3]),
#                 101: (Bottleneck, [3, 4, 23, 3]),
#                 152: (Bottleneck, [3, 8, 36, 3])}
# num_layers = 50
# style = 'pytorch'
# block_class, layers = resnet_spec[num_layers]
# model = PoseResNet(block_class, layers) 
# # measure
# macs_PoseResnet, params_PoseResnet = get_model_complexity_info(model, (3, 256, 256), as_strings=True, verbose=True)
# # 33.996 M, 100.000% Params, 7.267 GMac, 100.000% MACs

# # PoseMobileNet_normal
# model = PoseMobileNetV3(mode='normal')
# macs_PoseMobileNet_normal, params_PoseMobileNet_normal = get_model_complexity_info(model, (3, 256, 256), as_strings=True, verbose=True)
# # 2.86 M, 100.000% Params, 1.433 GMac, 100.000% MACs

# # PoseMobileNet_small
# model = PoseMobileNetV3(mode='small')
# macs_PoseMobileNet_small, params_PoseMobileNet_small = get_model_complexity_info(model, (3, 256, 256), as_strings=True, verbose=True)
# # 1.221 M, 100.000% Params, 0.648 GMac, 100.000% MACs

# # PoseMobileNet_large
# model = PoseMobileNetV3(mode='large')
# macs_PoseMobileNet_large, params_PoseMobileNet_large = get_model_complexity_info(model, (3, 256, 256), as_strings=True, verbose=True)
# #3.356 M, 100.000% Params, 1.55 GMac, 100.000% MACs