
"""
@author: Liang Deng
"""
#import os
import torch.nn as nn
import torch
from nets.CSPdarknet import C3, Conv, CSPDarknet
from SREencode import sre_layer

phi = 's'
depth_dict          = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
width_dict          = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]

base_channels       = int(wid_mul * 64)  # 64
base_depth          = max(round(dep_mul * 3), 1)  # 3
input_shape     = [640, 640]

class DarkNet(nn.Module):
    def __init__(self,num_classes, phi):
        super(DarkNet, self).__init__()
        depth_dict          = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3

        self.backbone   = CSPDarknet(base_channels, base_depth, phi, pretrained=True)
        
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")
        
        self.scale_rate = 256

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
    
    def forward(self, x):
        #  backbone

        feat1, feat2, feat3 = self.backbone(x)


        # 20, 20, 1024 -> 20, 20, 512
        P5          = self.conv_for_feat3(feat3)
        # 20, 20, 512 -> 40, 40, 512
        P5_upsample = self.upsample(P5)
        # 40, 40, 512 -> 40, 40, 1024
        P4          = torch.cat([P5_upsample, feat2], 1)
        # 40, 40, 1024 -> 40, 40, 512
        P4          = self.conv3_for_upsample1(P4)

        # 40, 40, 512 -> 40, 40, 256
        P4          = self.conv_for_feat2(P4)
        # 40, 40, 256 -> 80, 80, 256
        P4_upsample = self.upsample(P4)
        # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
        P3          = torch.cat([P4_upsample, feat1], 1)
        # 80, 80, 512 -> 80, 80, 256
        P3          = self.conv3_for_upsample2(P3)
        
        # 80, 80, 256 -> 40, 40, 256
        P3_downsample = self.down_sample1(P3)
        # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
        P4 = torch.cat([P3_downsample, P4], 1)
        # 40, 40, 512 -> 40, 40, 512
        P4 = self.conv3_for_downsample1(P4)

        # 40, 40, 512 -> 20, 20, 512
        P4_downsample = self.down_sample2(P4)
        # 20, 20, 512 cat 20, 20, 512 -> 20, 20, 1024
        P5 = torch.cat([P4_downsample, P5], 1)
        # 20, 20, 1024 -> 20, 20, 1024
        P5 = self.conv3_for_downsample2(P5)
        
        with torch.no_grad():
            
            if self.training:
                P5=sre_layer(P5,self.scale_rate)
            else:
                P5=sre_layer(P5,256)
                
            if self.training:
                P4=sre_layer(P4,self.scale_rate)
            else:
                P4=sre_layer(P4,256)    
                
            if self.training:
                P3=sre_layer(P3,self.scale_rate)
            else:
                P3=sre_layer(P3,256)

        tmp5 = nn.functional.adaptive_max_pool2d(P5, (1, 1))
        tmp4 = nn.functional.adaptive_max_pool2d(P4, (1, 1))
        tmp3 = nn.functional.adaptive_max_pool2d(P3, (1, 1))
        tmp = torch.cat([tmp5,tmp4,tmp3], 1)

        out = self.linear(tmp)
        return out
