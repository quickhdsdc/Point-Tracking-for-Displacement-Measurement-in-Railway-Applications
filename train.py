import numpy as np
import collections
import random
import sys
import os
import fnmatch
from datetime import datetime
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')
from ptflops import get_model_complexity_info
# import argparse
import torch
from torchvision import transforms
from torch import optim
from models.LightPointNet import LightPointNet
from models.PoseResNet import BasicBlock, Bottleneck, PoseResNet
from models.BlazePose import BlazePose
from models.ReceptionNet import ReceptionNet
from utils.dataset import read_data, KeypointsDataset, RandomHorizontalFlip, Normalize, ToTensor_reg, ToTensor, Resize
from utils.loss import L2JointLocationLoss, L1JointLocationLoss, JointsMSELoss, FocalLoss, SpatialSoftArgmax2dLoss
from utils.trainer import train
from utils import hparams_registry, misc
from utils.evaluation import evaluation
###############################################################################
#=================== arguments, parameters and environment ===================#
###############################################################################
# arguments can be given in comment line by argparse or in a dictionary like below
def run(args_dict):
    args = collections.namedtuple("args", args_dict.keys())(*args_dict.values())     
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print('Args:')
    for k, v in sorted(args_dict.items()):
        print('\t{}: {}'.format(k, v))
    
    if args.hparams:
        hparams = args.hparams
        # hparams.update(json.loads(args.hparams))  
    else:
        if args.hparams_seed == 0:
            hparams = hparams_registry.default_hparams(args.network)
        else:
            hparams = hparams_registry.random_hparams(args.network, misc.seed_hash(args.hparams_seed, args.trial_seed))
    
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    ###############################################################################
    #========================= prepare datasets ==================================#
    ###############################################################################
    ## read data
    if args.data_train == 'train256':
        data_train = './dataset/images/train256'
        label_train = './dataset/labels/train256'
        data_val = './dataset/images/valid256'
        label_val = './dataset/labels/valid256'
        data_test = './dataset/images/test256'
        label_test = './dataset/labels/test256'
    elif args.data_train == 'train256_aug':
        data_train = './dataset/images/train256_aug'
        label_train = './dataset/labels/train256_aug'
        data_val = './dataset/images/valid256_aug'
        label_val = './dataset/labels/valid256_aug'
        data_test = './dataset/images/test256_aug'
        label_test = './dataset/labels/test256_aug'
    elif args.data_train == 'train256_cornercases':
        data_train = './dataset/images/train256_cornercases'
        label_train = './dataset/labels/train256_cornercases'
        data_val = './dataset/images/valid256_cornercases'
        label_val = './dataset/labels/valid256_cornercases'
        data_test = './dataset/images/test256_cornercases'
        label_test = './dataset/labels/test256_cornercases'
    elif args.data_train == 'train256_new':
        data_train = './dataset/images/train256_new'
        label_train = './dataset/labels/train256_new'
        data_val = './dataset/images/valid256_new'
        label_val = './dataset/labels/valid256_new'
        data_test = './dataset/images/test256_new'
        label_test = './dataset/labels/test256_new'        
    elif args.data_train == 'train256_all':
        data_train = './dataset/images/train256_all'
        label_train = './dataset/labels/train256_all'
        data_val = './dataset/images/valid256_all'
        label_val = './dataset/labels/valid256_all'
        data_test = './dataset/images/test256_all'
        label_test = './dataset/labels/test256_all'  
    
    train_dataset=read_data(dirpath_img=data_train, dirpath_label=label_train)
    val_dataset=read_data(dirpath_img=data_val, dirpath_label=label_val)
    test_dataset=read_data(dirpath_img=data_test,dirpath_label=label_test)
    
    ## visualization to check the input data
    # from utils.dataset import show_images
    # show_images(train_dataset, range(4))   
    
    ## transform
    IMG_SIZE=args.image_size
    if args.loss == 'Integral' or args.loss == 'SpatialSoftArgmax2d' or (args.network == 'BlazePose' and args.continuous_training == True):
        ## when loss is integral loss, the ground truth should be coordinate of keypoints
        if hparams['transform_aug']:
            aug_tfms = transforms.Compose([Resize(img_size=IMG_SIZE), RandomHorizontalFlip(), Normalize(), ToTensor_reg()])
        else:
            aug_tfms = transforms.Compose([Resize(img_size=IMG_SIZE), Normalize(), ToTensor_reg()])   
        aug_tfms_test = transforms.Compose([Resize(img_size=IMG_SIZE), Normalize(), ToTensor_reg()])
    else:
        ## when loss is mse loss, the ground truth should be heatmap
        if hparams['transform_aug']:
            aug_tfms = transforms.Compose([Resize(img_size=IMG_SIZE), RandomHorizontalFlip(), Normalize(), ToTensor()])
        else:
            aug_tfms = transforms.Compose([Resize(img_size=IMG_SIZE), Normalize(), ToTensor()])   
        aug_tfms_test = transforms.Compose([Resize(img_size=IMG_SIZE), Normalize(), ToTensor()])    
    
    ## transfer dataset to pytorch format
    trainset = KeypointsDataset(train_dataset, transform=aug_tfms)
    valset = KeypointsDataset(val_dataset, transform=aug_tfms)
    testset = KeypointsDataset(test_dataset, train=False, transform=aug_tfms_test)
    
    BATCH_SIZE=hparams['batch_size']
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE)
    valid_loader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE)
   
    ###############################################################################
    #========================= build model and train =============================#
    ###############################################################################
    ## build model 
    if args.network == 'PoseResNet': 
        resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
                        34: (BasicBlock, [3, 4, 6, 3]),
                        50: (Bottleneck, [3, 4, 6, 3]),
                        101: (Bottleneck, [3, 4, 23, 3]),
                        152: (Bottleneck, [3, 8, 36, 3])}
        num_layers = hparams['num_layers']
        block_class, layers = resnet_spec[num_layers]
        model = PoseResNet(block_class, layers) 
        if hparams['pretrained']:
            model.init_weights(pretrained='./weights/pose_resnet_50_256x256.pth.tar')
        if args.continuous_training:
            model.load_state_dict(torch.load('./trained_models/PoseResNet50.pt'))
    
    # BlazePose has to be trained in two stages. First, train the heatmap branch. Second, train the regression branch
    elif args.network == 'BlazePose':
        if args.continuous_training == False:        
            model = BlazePose(train_mode=0)
        elif args.continuous_training == True:   
            model = BlazePose(train_mode=1)
            model.load_state_dict(torch.load('./trained_models/BlazePose_aug_stage0.pt'))
            ct = 0
            for child in model.children():
                ct += 1
                if ct >= 12:
                    # print(child)
                    for param in child.parameters():
                        param.requires_grad = False
    
    elif args.network == 'ReceptionNet':
        model = ReceptionNet()
        if args.continuous_training:
            model.load_state_dict(torch.load('./trained_models/ReceptionNet_stage0.pt')) 
                    
    elif args.network == 'LightPointNet_normal':
        model = LightPointNet(mode='normal')
        if args.continuous_training:
            model.load_state_dict(torch.load('./trained_models/LightPointNet_all.pt'))    
    
    elif args.network == 'LightPointNet_small':
        model = LightPointNet(mode='small')
        if args.continuous_training:
            model.load_state_dict(torch.load('./trained_models/PoseMobileNet_small.pt'))  
    
    elif args.network == 'LightPointNet_large':
        model = LightPointNet(mode='large')
        if args.continuous_training:
            model.load_state_dict(torch.load('./trained_models/PoseMobileNet_large.pt'))          
    
    model = model.to(device)
    
    ## loss
    if args.loss == 'Integral':
        # criterion = L2JointLocationLoss()
        criterion = L1JointLocationLoss()
    elif args.loss == 'MSE':    
        criterion = JointsMSELoss()
    elif args.loss == 'Focal':    
        criterion = FocalLoss()
    elif args.loss == 'SpatialSoftArgmax2d':
        criterion = SpatialSoftArgmax2dLoss()
    
    ## optimizer
    optimizer = optim.Adam(model.parameters(), lr=hparams['lr'],weight_decay=hparams['weight_decay'])
    ## checkpoints 
    if args.hparam_tuning:
        sub_dir = 'hparamstuning_' + args.network + '_' + args.loss + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M')
    else:
        sub_dir = args.network + '_' + args.loss + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M')
    save_dir = os.path.join(args.output_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    results_path = os.path.join(save_dir, 'results.jsonl')
    with open(results_path, 'a') as f:
        f.write(json.dumps(args_dict, sort_keys=True) + "\n")
        f.write(json.dumps(hparams, sort_keys=True) + "\n")       
        
    ## training
    train_losses, valid_losses = train(train_loader, valid_loader, model, 
                                       criterion, optimizer, hparams['lr_schedule'],                                       
                                       n_epochs=hparams['epoch'],
                                       saved_model=save_dir)
    
        ## complexity
    macs, num_params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, verbose=True)
    model_complexity = {'macs': macs, 'num_params': num_params}
    with open(results_path, 'a') as f:
        f.write(json.dumps(model_complexity) + "\n")
    ###############################################################################
    #============================== evaluation ===================================#
    ###############################################################################  
    n_Pw_list, n_Pr1_list, n_Pr2_list, n_Pwr1_list, model_list = [],[],[],[],[]
    listOfFiles = os.listdir(save_dir)
    pattern = "*.pt"
    for entry in listOfFiles:
        model_list.append(entry)
        if fnmatch.fnmatch(entry, pattern):     
            if (args.loss == 'MSE') or (args.loss == 'Focal'):
                output_type = 'heatmap'
            else:
                output_type = 'coordinates'
            if args.network == 'BlazePose' and args.continuous_training == True:
                output_type = 'coordinates_direct'
            args_eva = {'data_test': 'test256', # 'test256','test256_aug','test256_cornercases'  
                         'image_size': 256,
                         'output_type': output_type, # 'heatmap','coordinates'
                         'netwrok': args.network, # 'PoseResNet', 'BlazePose', 'ReceptionNet', 'LightPointNet_normal', 'LightPointNet_large', 'LightPointNet_small'
                         'model_path': os.path.join(save_dir, entry), #
                         'hparams': args.hparams,                   
                         'seed': 0
                        }
            
            n_Pw, n_Pr1, n_Pr2 = evaluation(args_eva)
            n_Pw_list.append(n_Pw)
            n_Pr1_list.append(n_Pr1)
            n_Pr2_list.append(n_Pr2)
            n_Pwr1_list.append(n_Pw+n_Pr1)            
    if len(n_Pwr1_list)>1:
        index = np.argmax(np.array(n_Pwr1_list)[:,0])
    else:
        index = 0    
    n_Pw = n_Pw_list[index]
    n_Pr1 = n_Pr1_list[index]
    n_Pr2 = n_Pr2_list[index]
    model_exp = model_list[index]
    results_eval = {'n_Pw': n_Pw.tolist(), 'n_Pr1': n_Pr1.tolist(), 'n_Pr2': n_Pr2.tolist(), 'model_best': model_exp}    
    with open(results_path, 'a') as f:
        f.write(json.dumps(results_eval) + "\n")
    
    losses = {'train_losses': train_losses, 'valid_losses': valid_losses}    
    with open(results_path, 'a') as f:
        f.write(json.dumps(losses) + "\n")
###############################################################################
#============================= run experiments ===============================#
###############################################################################
for network in ['BlazePose']: # 'LightPointNet_normal', 'PoseResNet','BlazePose','ReceptionNet','LightPointNet_large', 'LightPointNet_small'
    for loss in ['MSE']: # 'MSE', 'Focal', 'Integral', 'SpatialSoftArgmax2d'
        for i in range(1):
            args_dict = {'data_train': 'train256_aug', # 'train256','train256_aug','train256_cornercases','train256_new','train256_all'
                         'image_size': 256,
                         'loss': loss, # 'MSE', 'Focal', 'Integral', 'SpatialSoftArgmax2d'
                         'network': network, # 'PoseResNet', 'BlazePose', 'ReceptionNet', 'LightPointNet_normal', 'LightPointNet_large', 'LightPointNet_small'
                         'continuous_training': True, # True, False
                         'hparams_seed': 0,
                         'hparams': None,            
                         'trial_seed': 0,            
                         'seed': i,
                         'output_dir': './checkpoints/',
                         'hparam_tuning': False, # True, False
                         'num_workers': 8
                        }
            run(args_dict)