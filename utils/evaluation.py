import numpy as np
import collections
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

###############################################################################
#=================== arguments, parameters and environment ===================#
###############################################################################
# arguments can be given in comment line by argparse or in a dictionary like below
# args_dict = {'data_test': 'test256', # 'test256','test256_aug','test256_cornercases'  
#              'image_size': 256,
#              'output_type': 'coordinates', # 'heatmap','coordinates'
#              'netwrok': 'LightPointNet_normal', # 'PoseResNet', 'BlazePose', 'ReceptionNet', 'LightPointNet_normal', 'LightPointNet_large', 'LightPointNet_small'
#              'model_path': './trained_models/PoseResNet50.pt', #
#              'hparams': None,                   
#              'seed': 0
#             }
def evaluation(args_dict):
    args = collections.namedtuple("args", args_dict.keys())(*args_dict.values())     
        
    if args.hparams:
        hparams = args.hparams
        # hparams.update(json.loads(args.hparams))  
    else:
        hparams = hparams_registry.default_hparams(args.netwrok)
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    ###############################################################################
    #========================= prepare datasets ==================================#
    ###############################################################################
    ## read data
    if args.data_test == 'test256':
        data_test = './dataset/images/test256'
        label_test = './dataset/labels/test256'
    elif args.data_test == 'test256_aug':
        data_test = './dataset/images/test256_aug'
        label_test = './dataset/labels/test256_aug'
    elif args.data_test == 'test256_cornercases':
        data_test = './dataset/images/test256_cornercases'
        label_test = './dataset/labels/test256_cornercases'
    
    test_dataset=read_data(dirpath_img=data_test,dirpath_label=label_test)
    
    ## transform
    IMG_SIZE=args.image_size
    if args.output_type == 'coordinates' or args.output_type == 'coordinates_direct':
        aug_tfms_test = transforms.Compose([Resize(img_size=IMG_SIZE), Normalize(), ToTensor_reg()])
    elif args.output_type == 'heatmap': 
        aug_tfms_test = transforms.Compose([Resize(img_size=IMG_SIZE), Normalize(), ToTensor()])    
    
    testset = KeypointsDataset(test_dataset, train=False, transform=aug_tfms_test)
    BATCH_SIZE=hparams['batch_size']
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
    
    ###############################################################################
    #================= read trained models and make interference =================#
    ############################################################################### 
    ## build model 
    if args.netwrok == 'PoseResNet': 
        resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
                        34: (BasicBlock, [3, 4, 6, 3]),
                        50: (Bottleneck, [3, 4, 6, 3]),
                        101: (Bottleneck, [3, 4, 23, 3]),
                        152: (Bottleneck, [3, 8, 36, 3])}
        num_layers = hparams['num_layers']
        block_class, layers = resnet_spec[num_layers]
        model = PoseResNet(block_class, layers) 
        model.load_state_dict(torch.load(args.model_path))
    
    elif args.netwrok == 'BlazePose':
        if args.output_type == 'coordinates' or args.output_type == 'coordinates_direct':
            model = BlazePose(train_mode=1)
        elif args.output_type == 'heatmap': 
            model = BlazePose(train_mode=0)
        model.load_state_dict(torch.load(args.model_path))
    
    elif args.netwrok == 'ReceptionNet':
        model = ReceptionNet()
        model.load_state_dict(torch.load(args.model_path)) 
                    
    elif args.netwrok == 'LightPointNet_normal':
        model = LightPointNet(mode='normal')
        model.load_state_dict(torch.load(args.model_path))    
    
    elif args.netwrok == 'LightPointNet_small':
        model = LightPointNet(mode='small')
        model.load_state_dict(torch.load(args.model_path))  
    
    elif args.netwrok == 'LightPointNet_large':
        model = LightPointNet(mode='large')
        model.load_state_dict(torch.load(args.model_path))          
    
    model = model.to(device)
    predictions = predict(test_loader, model)
    if args.output_type == 'heatmap':
        keypoints_pred, maxvals=get_max_preds(predictions)
        keypoints_pred=keypoints_pred*4 # image is 256x256, heatmap is 64x64
        keypoints_pred=np.reshape(keypoints_pred,[-1,6])
    elif args.output_type == 'coordinates':
        keypoints_pred = get_joint_location_result(torch.tensor(predictions).to(device))
        keypoints_pred=np.reshape(keypoints_pred,[-1,6])
    elif args.output_type == 'coordinates_direct':  
        keypoints_pred=np.reshape(predictions,[-1,6])
        keypoints_pred=keypoints_pred*256 # image_size 256
        
        
    test_predictions=pd.DataFrame(keypoints_pred) 
    test_predictions=test_predictions.rename(columns={0:"ref_w_x", 1:"ref_w_y", 2:"ref_r1_x", 3:"ref_r1_y", 4:"ref_r2_x", 5:"ref_r2_y"})
    test_predictions['image']=test_dataset['image']
    
    deviation_Pr1 = abs((test_predictions["ref_r1_x"] - test_dataset["ref_r1_x"]).dropna().values)
    deviation_Pw = abs((test_predictions["ref_w_x"] - test_dataset["ref_w_x"]).dropna().values)
    deviation_Pr2 = abs((test_predictions["ref_r2_x"] - test_dataset["ref_r2_x"]).dropna().values)
    
    num = len(deviation_Pr1)
    bins = [0, 1.5, 5.5, 20.5, 256]
    n_Pw,_,_ = plt.hist(deviation_Pw, bins)
    n_Pr1,_,_ = plt.hist(deviation_Pr1, bins)
    n_Pr2,_,_ = plt.hist(deviation_Pr2, bins)
    plt.close()
      
    return n_Pw, n_Pr1, n_Pr2

