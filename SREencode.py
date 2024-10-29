
"""
@author: Liang Deng
"""

#import matplotlib.pyplot as plt
#import cv2
#import matplotlib.image as mpimg
#import matplotlib.cm as cm
import torch
import numpy as np



def sre_encoding(img,circ):    
    image_f = np.fft.fft2(img)
    image_f = np.fft.fftshift(image_f)
    f_a = np.imag(image_f)
    f_b = np.real(image_f)
    filtered_a = f_a*circ
    filtered_b = f_b*circ
    filtered_f = filtered_b + filtered_a*1j
    image_feat=np.fft.ifft2(filtered_f)
    return abs(image_feat)

def disk(r,bound):
    f = np.zeros((bound,bound))
    for i in range(bound):
        for j in range(bound):
            if (np.power(i-bound/2,2) + np.power(j-bound/2,2)) < r**2:
                f[i,j]=1
    return f

def sre_layer(tensor,scale_rate):
    bs,ch,h,w = tensor.shape
    P = np.zeros((bs,ch,h,w))
    radius = radius_cal(scale_rate,h)
    circ1 = disk(radius,h)   
    tensor = tensor.cpu()
    numpy_array = tensor.detach().numpy()   
    for i in range(bs):
        for j in range(ch):
            feature1 = numpy_array[i,j,:,:]
            P[i,j,:,:]=sre_encoding(feature1,circ1)

    P = torch.Tensor(P).cuda(0)

    return P
        


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)

def radius_cal(scale_rate,bound):
    """
    scale_rate:range from 64 to 256
    """
    t = interpolant((scale_rate-64)/192)
    radius = 5+t*(bound/2-5)
    return radius

