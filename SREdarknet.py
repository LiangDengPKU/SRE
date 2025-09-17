#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dengl
"""

import torch.nn as nn
import torch
from nets.CSPdarknet import C3, Conv, CSPDarknet
from SREencoding import SRELayer
import numpy as np


class SREDarkNet(nn.Module):
    def __init__(self,num_classes, phi):
        super(SREDarkNet, self).__init__()
        depth_dict          = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]
        base_channels       = int(wid_mul * 64)  
        base_depth          = max(round(dep_mul * 3), 1)  
        self.backbone   = CSPDarknet(base_channels, base_depth, phi, pretrained=True)        
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")       
        self.conv_for_feat3         = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1    = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)
        self.conv_for_feat2         = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2    = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)
        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1  = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)
        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2  = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)
        self.linear = nn.Sequential(
                                        nn.Flatten(),
                                        nn.Linear(896,  1024),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(1024, num_classes)
                                    )
        self.sre_layer = SRELayer(input_tensor_shape=256, one_cell_fov=64, steepness=10.0)
    
    def forward(self, x, scale_rate=None):
        if scale_rate is None:
            scale_rate = torch.tensor(256.0, device=x.device, dtype=torch.float32)  
        elif isinstance(scale_rate, (int, float, np.integer)):
            scale_rate = torch.tensor(float(scale_rate), device=x.device, dtype=torch.float32)

        feat1, feat2, feat3 = self.backbone(x)
        # 20*20
        P5          = self.conv_for_feat3(feat3)
        # 20*20
        P5_upsample = self.upsample(P5)
        # 40*40
        P4          = torch.cat([P5_upsample, feat2], 1)
        # 40*40
        P4          = self.conv3_for_upsample1(P4)
        # 40*40
        P4          = self.conv_for_feat2(P4)
        # 40*40
        P4_upsample = self.upsample(P4)
        # 80*80
        P3          = torch.cat([P4_upsample, feat1], 1)
        # 80*80
        P3          = self.conv3_for_upsample2(P3)
        
        # 80*80
        P3_downsample = self.down_sample1(P3)
        # 40*40
        P4 = torch.cat([P3_downsample, P4], 1)
        # 40*40
        P4 = self.conv3_for_downsample1(P4)

        # 40*40, 512 -> 20, 20, 512
        P4_downsample = self.down_sample2(P4)
        # 20*20
        P5 = torch.cat([P4_downsample, P5], 1)
        # 20*20
        P5 = self.conv3_for_downsample2(P5)
        
        P5 = self.sre_layer(P5, scale_rate)
        P4 = self.sre_layer(P4, scale_rate)
        P3 = self.sre_layer(P3, scale_rate)
        
        tmp5 = nn.functional.adaptive_max_pool2d(P5, (1, 1))
        tmp4 = nn.functional.adaptive_max_pool2d(P4, (1, 1))
        tmp3 = nn.functional.adaptive_max_pool2d(P3, (1, 1))
        tmp = torch.cat([tmp5, tmp4, tmp3], 1)
        out = self.linear(tmp)
        return out
