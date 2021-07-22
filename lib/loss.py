import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

###############################################################################
#============================== loss =========================================#
###############################################################################
def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)

# def _focal_loss(pred, gt):
#     pos_inds = gt.eq(1)
#     neg_inds = gt.lt(1)

#     neg_weights = torch.pow(1 - gt[neg_inds], 4)

#     loss = 0
#     pos_pred = pred[pos_inds]
#     neg_pred = pred[neg_inds]

#     pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
#     neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

#     num_pos  = pos_inds.float().sum()
#     pos_loss = pos_loss.sum()
#     neg_loss = neg_loss.sum()

#     if pos_pred.nelement() == 0:
#         loss = loss - neg_loss
#     else:
#         loss = loss - (pos_loss + neg_loss) / num_pos
#     return loss

def _focal_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    
    neg_weights = torch.pow(1 - gt, 4)
    
    loss = 0
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    
    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
  def __init__(self,focal_loss=_focal_loss):
      super(FocalLoss, self).__init__()
      self.focal_loss  = focal_loss

  def forward(self, pred, gt):
      batch_size = pred.size(0)
      num_joints = pred.size(1)
      hp_size = pred.size(3)   
      # pred=pred.view(batch_size*num_joints*hp_size*hp_size)
      # gt=gt.view(batch_size*num_joints*hp_size*hp_size)
      
      pred = _sigmoid(pred)
      # gt = _sigmoid(gt)
      
      focal_loss = 0
      focal_loss += self.focal_loss(pred, gt)
    
      loss = focal_loss/num_joints
      return loss
##############################################################################
class JointsMSELoss(nn.Module):
    def __init__(self):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)

    def forward(self, output, target):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()

            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        return loss / num_joints