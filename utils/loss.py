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

##############################################################################
## intergral loss
def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

def weighted_mse_loss(output, target, size_average):
    out = (output - target) ** 2
    if size_average:
        return out.sum() / len(output)
    else:
        return out.sum()

def weighted_l1_loss(output, target, size_average):
    out = torch.abs(output - target)
    if size_average:
        return out.sum() / len(output)
    else:
        return out.sum()

def generate_3d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim, z_dim):
    assert isinstance(heatmaps, torch.Tensor)

    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, z_dim, y_dim, x_dim))

    accu_x = heatmaps.sum(dim=2)
    accu_x = accu_x.sum(dim=2)
    accu_y = heatmaps.sum(dim=2)
    accu_y = accu_y.sum(dim=3)
    accu_z = heatmaps.sum(dim=3)
    accu_z = accu_z.sum(dim=3)
    
    device = torch.device('cuda')
    accu_x = accu_x * torch.arange(float(x_dim)).to(device)
    accu_y = accu_y * torch.arange(float(y_dim)).to(device)
    accu_z = accu_z * torch.arange(float(z_dim)).to(device)
    # accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(x_dim).type(torch.cuda.FloatTensor), devices=[accu_x.device.index])[0]
    # accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(y_dim).type(torch.cuda.FloatTensor), devices=[accu_y.device.index])[0]
    # accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(z_dim).type(torch.cuda.FloatTensor), devices=[accu_z.device.index])[0]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    return accu_x, accu_y, accu_z


def softmax_integral_tensor(preds, num_joints, hm_width, hm_height, hm_depth):
    # global soft max
    preds = preds.reshape((preds.shape[0], num_joints, -1))
    preds = F.softmax(preds, 2)

    # integrate heatmap into joint location
    x, y, _ = generate_3d_integral_preds_tensor(preds, num_joints, hm_width, hm_height, hm_depth)

    x = x / float(hm_width) - 0.5
    y = y / float(hm_height) - 0.5
    preds = torch.cat((x, y), dim=2)
    preds = preds.reshape((preds.shape[0], num_joints * 2))
    return preds

class L2JointLocationLoss(nn.Module):
    def __init__(self):
        super(L2JointLocationLoss, self).__init__()

    def forward(self, output, target):
        num_joints = output.size(1)
        gt_joints = target
        
        hm_width = output.shape[2]
        hm_height = output.shape[3]
        hm_depth = 1

        pred_jts = softmax_integral_tensor(output, num_joints, hm_width, hm_height, hm_depth)

        _assert_no_grad(gt_joints)
        return weighted_mse_loss(pred_jts, gt_joints, False)


class L1JointLocationLoss(nn.Module):
    def __init__(self):
        super(L1JointLocationLoss, self).__init__()

    def forward(self, output, target):
        num_joints = output.size(1)
        gt_joints = target

        hm_width = output.shape[2]
        hm_height = output.shape[3]
        hm_depth = 1

        pred_jts = softmax_integral_tensor(output, num_joints, hm_width, hm_height, hm_depth)

        _assert_no_grad(gt_joints)
        return weighted_l1_loss(pred_jts, gt_joints, False)

##############################################################################
## SpatialSoftArgmax2d
def create_meshgrid(x):
    _, _, height, width = x.shape
    _device, _dtype = x.device, x.dtype
    xs = torch.linspace(0, width-1, width, device=_device, dtype=_dtype)
    ys = torch.linspace(0, height-1, height, device=_device, dtype=_dtype)
    return torch.meshgrid(ys, xs)  # pos_y, pos_x

class SpatialSoftArgmax2dLoss(nn.Module):
    def __init__(self):
        super(SpatialSoftArgmax2dLoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.eps: float = 1e-6

    def forward(self, output, target):
        batch_size = output.size(0)
        num_joints = output.size(1)
        x = output.view(batch_size, num_joints, -1)
        # compute softmax with max substraction trick
        exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
        exp_x_sum = 1.0 / (exp_x.sum(dim=-1, keepdim=True) + self.eps)
        # create coordinates grid
        pos_y, pos_x = create_meshgrid(output)
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        # compute the expected coordinates
        expected_y = torch.sum((pos_y * exp_x) * exp_x_sum, dim=-1, keepdim=True)
        expected_x = torch.sum((pos_x * exp_x) * exp_x_sum, dim=-1, keepdim=True)
        pred_jts = torch.cat([expected_x, expected_y], dim=-1)
        pred_jts = pred_jts.view(batch_size, -1)
        gt_jts = target.view(batch_size, -1)
        _assert_no_grad(gt_jts)
        return weighted_mse_loss(pred_jts, gt_jts, False)