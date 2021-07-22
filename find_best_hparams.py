import collections
import os, fnmatch
import json
from utils import misc
import numpy as np

args_dict = {'data_train': 'train256', # 'train256','train256_aug','train256_cornercases'  
             'image_size': 256,
             'transform_aug': False, # True, False
             'loss': 'MSE', # 'MSE', 'Focal', 'Integral', 'SpatialSoftArgmax2d'
             'network': 'LightPointNet_normal', # 'PoseResNet', 'BlazePose', 'ReceptionNet', 'LightPointNet_normal', 'LightPointNet_large', 'LightPointNet_small'
             'continuous_training': False, # True, False
             'hparams_seed': 0,
             'hparams': None,            
             'trial_seed': 0,            
             'seed': 1,
             'output_dir': './checkpoints/',
             'hparam_tuning': True, # True, False
             'num_workers': 8
            }
args = collections.namedtuple("args", args_dict.keys())(*args_dict.values())

listOfFolder = os.listdir(args.output_dir)
# pattern = "hparamstuning_" + args.network + "*"
pattern = args.network + "*"
acc_best = 0
acc_good = []
args_good = []
hparams_good = []
for folder in listOfFolder:
    if fnmatch.fnmatch(folder, pattern):
        folderpath = os.path.join(args.output_dir, folder)
        filepath = os.path.join(folderpath, 'results.jsonl')
        with open(filepath, 'rb') as f:
            json_list = list(f)
        result = json.loads(json_list[3])
        n_Pw = np.array(result['n_Pw'])
        n_Pr1 = np.array(result['n_Pr1'])
        n_Pr2 = np.array(result['n_Pr2'])
        acc_temp = (n_Pw[0]/np.sum(n_Pw) + n_Pr1[0]/np.sum(n_Pr1) + n_Pr2[0]/np.sum(n_Pr2))/3        
        if acc_temp>acc_best:
            acc_best = acc_temp
            args_best = json.loads(json_list[0])
            hparams_best = json.loads(json_list[1])   
        if acc_temp>0.5:
            acc_good.append(acc_temp)
            args_good.append(json.loads(json_list[0]))
            hparams_good.append(json.loads(json_list[1]))
misc.print_row(hparams_best, colwidth=12)
misc.print_row([hparams_best[key] for key in hparams_best.keys()], colwidth=12)
misc.print_row(args_best, colwidth=12)
misc.print_row([args_best[key] for key in args_best.keys()], colwidth=12)       
## select top5 hparams
idx_rank3=sorted(range(len(acc_good)), reverse=True, key=lambda k: acc_good[int(k)])     
hparams = [hparams_good[i] for i in idx_rank3[:3]]



        