import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from utils.dataset import show_keypoints
from utils.loss import softmax_integral_tensor

device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def predict(data_loader, model):
    '''
    Predict keypoints
    Args:
        data_loader (DataLoader): DataLoader for Dataset
        model (nn.Module): trained model for prediction.
    Return:
        predictions (array-like): keypoints in float (no. of images x keypoints).
    '''
    
    model.eval() # prep model for evaluation

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(batch['image'].to(device)).cpu().numpy()
            if i == 0:
                predictions = output
            else:
                predictions = np.vstack((predictions, output))
    
    return predictions

def get_max_preds(batch_heatmaps):
    '''
    get predictions from heatmaps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    
    return preds, maxvals

IMAGE_SIZE = 256
# define result
def get_joint_location_result(batch_heatmaps):
    '''
    get predictions from heatmaps using integral loss
    '''     
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    hm_width = batch_heatmaps.shape[3]    
    hm_height = hm_width
    hm_depth = 1

    pred_jts = softmax_integral_tensor(batch_heatmaps, num_joints, hm_width, hm_height, hm_depth)
    coords = pred_jts.detach().cpu().numpy()
    coords = coords.astype(float)
    coords = coords.reshape((batch_size, num_joints, 2))
    # project to original image size
    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * IMAGE_SIZE
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * IMAGE_SIZE
    return coords

def save_output(df):
   for idx in range(len(df)):
       image = df.loc[idx, 'image']
       image = Image.open(image)      
       keypoints = df.loc[idx].drop('image').values.reshape(-1, 2)
       plt.figure()
       plt.tight_layout()
       show_keypoints(image, keypoints)
       img_name=df.loc[idx,'image']
       img_name=img_name.split('/')[-1]
       img_path='./output/'+img_name
       plt.savefig(img_path)
       plt.close()