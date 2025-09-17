#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Liang Deng
"""

import torch
import torch.nn as nn
import torch.fft as fft

ORIGINAL_DIMENSION = 256
def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)

class SRELayer(nn.Module):
    def __init__(self, input_tensor_shape=ORIGINAL_DIMENSION, one_cell_fov=64, steepness=10.0):
        super().__init__()
        self.input_tensor_shape = input_tensor_shape
        self.one_cell_fov = one_cell_fov
        self.steepness = steepness  
        self.offset1 = nn.Parameter(torch.tensor(0.0))
        self.offset2 = nn.Parameter(torch.tensor(0.0))

    def radius_cal(self, scale_rate, bound):
        r0 = 2.439
        upper_dimention = self.input_tensor_shape - self.offset1
        lower_dimention = self.one_cell_fov + self.offset2
        denom = upper_dimention - lower_dimention
        t = (scale_rate - lower_dimention) / (denom + 1e-8)
        t = torch.clamp(t, 0.0, 1.0)
        t_interp = interpolant(t)
        radius = r0 + t_interp * (bound / 4 - r0)
        return radius

    def create_circular_mask(self, r, bound, device):
        y, x = torch.meshgrid(
            torch.arange(bound, device=device, dtype=torch.float32),
            torch.arange(bound, device=device, dtype=torch.float32),
            indexing='ij'
        )
        center = bound / 2.0
        dist_sq = (x - center) ** 2 + (y - center) ** 2
        dist = torch.sqrt(dist_sq + 1e-8)  

        mask = torch.sigmoid(self.steepness * (r - dist))
        return mask

    def sre_encoding(self, img, circ):
        image_f = fft.fft2(img)
        image_f = fft.fftshift(image_f)
        filtered_f = image_f * circ  
        image_feat = fft.ifft2(fft.ifftshift(filtered_f))
        return torch.abs(image_feat)

    def sre_encoding_batch(self, img_batch, circ):
        """
        img_batch: [B, C, H, W]
        circ: [H, W]
        """
        image_f = fft.fft2(img_batch)                    # [B, C, H, W]
        image_f = fft.fftshift(image_f, dim=(-2, -1))    
        filtered_f = image_f * circ                      # [B, C, H, W] * [H, W] → 广播
        image_feat = fft.ifft2(fft.ifftshift(filtered_f, dim=(-2, -1)))
        return torch.abs(image_feat)                     # [B, C, H, W]
    
    def forward(self, tensor, scale_rate):
        bs, ch, h, w = tensor.shape
        assert h == w, "square tensor only"   
        radius = self.radius_cal(scale_rate, h)
        circ = self.create_circular_mask(radius, h, tensor.device)
        output = self.sre_encoding_batch(tensor, circ)    
        return output

