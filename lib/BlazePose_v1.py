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
from lib.mobilenetv3 import MobileNetV3

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BlazePose(nn.Module):

    def __init__(self, train_mode=0, inplanes=16):
        self.deconv_with_bias = False
        self.train_mode = train_mode
        self.inplanes = inplanes
        self.block = Bottleneck

        super(BlazePose, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(self.block, self.inplanes, 3, stride=2)
        self.layer2 = self._make_layer(self.block, self.inplanes*2, 3, stride=2)
        self.layer3 = self._make_layer(self.block, self.inplanes*4, 3, stride=2)
        self.layer4 = self._make_layer(self.block, self.inplanes*8, 3, stride=2)   
        
        # heatmap branch  
        self.conv6 = nn.Conv2d(self.inplanes*32, self.inplanes*8, kernel_size=1, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)        
        kernel, padding, output_padding = self._get_deconv_cfg(4, 0)
        self.conv6_dec = nn.ConvTranspose2d(self.inplanes*8, self.inplanes*8, kernel_size=kernel, stride=2, padding=padding, 
                                         output_padding=output_padding, bias=self.deconv_with_bias)
        self.relu = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(self.inplanes*16, self.inplanes*8, kernel_size=1, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)        
        self.conv7_dec = nn.ConvTranspose2d(self.inplanes*8, self.inplanes*8, kernel_size=kernel, stride=2, padding=padding, 
                                         output_padding=output_padding, bias=self.deconv_with_bias)    
        self.relu = nn.ReLU(inplace=True)
        
        self.conv8 = nn.Conv2d(self.inplanes*8, self.inplanes*8, kernel_size=1, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv8_dec = nn.ConvTranspose2d(self.inplanes*8, self.inplanes*8, kernel_size=kernel, stride=2, padding=padding, 
                                         output_padding=output_padding, bias=self.deconv_with_bias)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv9 = nn.Conv2d(self.inplanes*4, self.inplanes*8, kernel_size=1, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv9_dec = nn.ConvTranspose2d(self.inplanes*8, self.inplanes*8, kernel_size=kernel, stride=2, padding=padding, 
                                         output_padding=output_padding, bias=self.deconv_with_bias)    
        self.relu = nn.ReLU(inplace=True)
        
        self.heatmap = nn.Conv2d(
            in_channels=self.inplanes*8,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # regression branch
        self.conv10 = nn.Conv2d(self.inplanes*2, self.inplanes*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)        

        self.conv11 = nn.Conv2d(self.inplanes*4, self.inplanes*8, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.conv12 = nn.Conv2d(self.inplanes*8, self.inplanes*12, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)  
        
        self.conv13 = nn.Conv2d(self.inplanes*12, self.inplanes*12, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)  

        self.conv14 = nn.Conv2d(self.inplanes*12, self.inplanes*12, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)         

        self.avg15 = nn.AvgPool2d(2)
        self.linear15 = nn.Linear(self.inplanes*12, 6)    


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        inplanes = self.inplanes
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * 4       
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)
    
    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        y1 = self.conv1(x)
        y1 = self.bn1(y1)
        y1 = self.relu(y1)

        y2 = self.layer1(y1)       
        y3 = self.layer2(y2)
        y4 = self.layer3(y3)        
        y5 = self.layer4(y4)
        
        # heatmap branch
        y6 = self.conv6(y5)
        y6 = self.relu(y6)
        
        y7a = self.conv6_dec(y6)
        y7a = self.relu(y7a)         
        y7b = self.conv7(y4)
        y7b = self.relu(y7b)
        y7 = y7a + y7b
                 
        y8a = self.conv7_dec(y7)
        y8a = self.relu(y8a)         
        y8b = self.conv8(y3)
        y8b = self.relu(y8b)
        y8 = y8a + y8b

        y9a = self.conv8_dec(y8)
        y9a = self.relu(y9a)         
        y9b = self.conv9(y2)
        y9b = self.relu(y9b)
        y9 = y9a + y9b
    
        y_hp = self.heatmap(y9)
        
        # regression branch
        y10 = y2 + y9
        y11a = self.conv10(y10)
        y11a = self.relu(y11a)
        y11 = y3 + y11a
        y12a = self.conv11(y11)
        y12a = self.relu(y12a)
        y12 = y4 + y12a   
        y13a = self.conv12(y12)
        y13a = self.relu(y13a)
        y13 = y5 + y13a       
        y14 = self.conv13(y13)
        y14 = self.relu(y14)   
        y15 = self.conv14(y14)
        y15 = self.relu(y15) 
        y16 = self.avg15(y15)  
        y16 = y16.view(y16.size(0),self.inplanes*16)
        y_c = self.linear15(y16)
        
        y = [y_hp, y_c]
        if self.train_mode == 0:
            return y[0]
        else:
            return y[1]

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





