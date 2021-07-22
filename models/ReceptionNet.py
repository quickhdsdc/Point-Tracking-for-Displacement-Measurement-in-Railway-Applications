# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict
from models._inceptionv4 import Inceptionv4_stem, SepconvResidual, Conv2d, SeparableConv2d

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class ReceptionNet(nn.Module):
    def __init__(self):
        self.deconv_with_bias = False
        super(ReceptionNet, self).__init__()
        
        self.inception=Inceptionv4_stem()
        # build the revised prediction block
        self.branch_a_mp = nn.MaxPool2d(2, stride=2, padding=0)
        self.mixed_a_conv = Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.branch_a_sepRe = SepconvResidual(256, 256, 3, 1)
        
        self.branch_b_mp = nn.MaxPool2d(2, stride=2, padding=0)
        self.branch_b_sepRe = SepconvResidual(256, 256, 3, 1) 
        self.branch_b_sepRe1 = SepconvResidual(256, 256, 3, 1) 
        self.branch_b_upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.branch_c_upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.branch_c_sepRe = SepconvResidual(512, 256, 3, 1) 
     
        self.deconv =  nn.ConvTranspose2d(256, 256, 4, 4, 0, bias=False)
        # self.deconv = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
      
        self.final_layer = nn.Conv2d(256, 3, 1, 1, 0)

    def forward(self, x):
        x = self.inception(x) # 16x16x512
        x_a = self.branch_a_mp(x)
        x_a = self.mixed_a_conv(x_a)
        x_a = self.branch_a_sepRe(x_a) # 8x8x256
        
        x_b = self.branch_b_mp(x_a)
        x_b = self.branch_b_sepRe(x_b)
        x_b = self.branch_b_sepRe1(x_b) # 4x4x256
        x_b = self.branch_b_upsample(x_b) # 8x8x256

        x_c = x_a + x_b
        x_c = self.branch_c_upsample(x_c) # 16x16x256
        x_c = x_c + self.branch_c_sepRe(x)
        
        x = self.deconv(x_c) # 64x64x256
        x = self.final_layer(x) # 64x64x3
        
        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            # pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            checkpoint = torch.load(pretrained)
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained))
            model_dict = self.state_dict()
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict and k != "final_layer.bias" and k != "final_layer.weight"}
            model_dict.update(state_dict)     
            self.load_state_dict(model_dict, strict=False)
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')





