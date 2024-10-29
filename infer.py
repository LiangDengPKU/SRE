
"""
@author: Liang Deng
"""

import os
os.chdir('./SRE-main')
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import measure
import openslide
from skimage.filters import threshold_otsu

from openslide.deepzoom import DeepZoomGenerator
import pandas as pd
#import glob
import time
#from scipy.ndimage import gaussian_filter
from SREdarknet import DarkNet
import torch
import torch.nn as nn

"""
Identification of Tumor Cell Clusters in Histopathology Images Using a Deep Learning Model with Scale Rate Encoding
specifically telangiectatic osteosarcoma (malignant) and aneurysmal bone cysts (benign).
"""


PATH='./whole_silde/14_05_13.svs'
WIDTH = 256
HEIGHT = 256
N_CLASSES = 3
BATCHSIZE = 8
model_path = "./logs/ep300-loss0.024-val_loss0.040.pth"
model_t = DarkNet(3, "s")
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_t.load_state_dict(torch.load(model_path, map_location=device))
cuda = True
model_t = model_t.eval()
if cuda:
    model_t = nn.DataParallel(model_t)
    model_t = model_t.cuda()



def linear_normalize(img_):
    img_ln = img_/255.0
    return img_ln    
 
def minmax_normalize(img_):
    img_mn = cv2.normalize(img_, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img_mn 


def find_tissue_patches(slide_path_):
    with openslide.open_slide(slide_path_) as slide:
        thumbnail = slide.get_thumbnail((slide.dimensions[0] / HEIGHT, slide.dimensions[1] / WIDTH))
    
    thumbnail_grey = np.array(thumbnail.convert('L')) # convert to grayscale
    thresh = threshold_otsu(thumbnail_grey)
    binary = thumbnail_grey > thresh

    
    samples = pd.DataFrame(pd.DataFrame(binary).stack())
    samples['is_tissue'] = ~samples[0]
    samples.drop(0, axis=1, inplace=True)

    samples = samples[samples.is_tissue == True] # remove patches with no tissue
    samples['tile_loc'] = list(samples.index)
    samples.reset_index(inplace=True, drop=True)
    return samples,binary



def pre_img_batch(imgs_,n_):   
    batch_images_tf_ = imgs_
    batch_images_tf_ = np.array(batch_images_tf_).reshape((n_, 3, WIDTH, HEIGHT))
    batch_images_tf_ = linear_normalize(batch_images_tf_) 
    batch_images_tf_ = np.array(batch_images_tf_,dtype=np.float32)
    return batch_images_tf_



def get_rid_off_fp(mask_,thres_,label_,h2_,w2_):    
    all_label, num= measure.label(mask_, return_num=True, connectivity=2) 
    #print(all_label)
    props = measure.regionprops(all_label)
    numPix = []
    index = []
    for ia in range(len(props)):
        if props[ia].area >= thres_:
            numPix += [props[ia].area]
            index.append(ia)
    mask1 = np.zeros((h2_, w2_))
    if len(numPix) != 0:
        maxnum = max(numPix)
        print('max area:',maxnum)

    for i in range(len(index)):
        label1 = props[index[i]].coords
        for j in range(len(label1)):
            mask1[label1[j][0],label1[j][1]]=label_ 
    mask1 = np.array(mask1,np.uint8)
    return mask1    



def bm_color_encode(mask_thumb_all2_):
    h3,w3 = np.shape(mask_thumb_all2_)
    mask_thumb_all3 = np.zeros((h3,w3,3),dtype = np.uint8)

    r = np.zeros((h3,w3),dtype = np.uint8)
    g = np.zeros((h3,w3),dtype = np.uint8)
    b = np.zeros((h3,w3),dtype = np.uint8)

    r[mask_thumb_all2_ == 1] = 1
    g[mask_thumb_all2_ == 1] = 199
    b[mask_thumb_all2_ == 1] = 140

    r[mask_thumb_all2_ == 2] = 138
    g[mask_thumb_all2_ == 2] = 43
    b[mask_thumb_all2_ == 2] = 226

    mask_thumb_all3[:,:,0] = r
    mask_thumb_all3[:,:,1] = g
    mask_thumb_all3[:,:,2] = b

    mask_thumb_all3[mask_thumb_all3==0]=255
    return mask_thumb_all3
#####################################################################################
#####################################################################################


"""
normal tissue: 0
benign: 1
malignant: 2
hemo: 0
normal bone: 0

"""


def infer(all_tissue_samples_,slide_):
    start_time = time.time()    

    benign_id = []
    malignant_id = []
    #bg_id = []

    N = len(all_tissue_samples_)

    tiles = DeepZoomGenerator(slide_, tile_size=HEIGHT, overlap=0, limit_bounds=False)

    for i in range(N//BATCHSIZE):  
        imgs=[]
        for j in range(i*BATCHSIZE,i*BATCHSIZE+BATCHSIZE):
                         
            b,a=all_tissue_samples_.iat[j,1]
            img_1 =tiles.get_tile(tiles.level_count-1, (a,b))
            img_1 = np.array(img_1)
            imgs.append(img_1)
        
        batch_images_tf = pre_img_batch(imgs,BATCHSIZE)
        batch_images_tf = torch.Tensor(batch_images_tf)

        batch_images_tf  = batch_images_tf.cuda(0)
        pred_mask = model_t(batch_images_tf)
        pred_mask = pred_mask.cpu()
        pred_mask = pred_mask.detach().numpy()         
        pred_mask1 = np.argmax(pred_mask,axis=1)
        
        for k in range(BATCHSIZE):        
            pred_mask2 = pred_mask1[k]
            if pred_mask2 == 1:
                benign_id.append(i*BATCHSIZE+k)
            if pred_mask2 == 2:
                malignant_id.append(i*BATCHSIZE+k)
                
        if i%100 == 0:
            print('proceeding tiles:',i*8)
        
    ends = N%BATCHSIZE
    N1 = N-ends
        
    if ends != 0:
        remain_imgs = []
        for i in range(ends):
            n1 = N - i -1
            print('ends:',n1)
            b,a = all_tissue_samples_.iat[n1,1]
            img_1 =tiles.get_tile(tiles.level_count-1, (a,b))
            img_1 = np.array(img_1)
            remain_imgs.append(img_1)
               
        batch_images_tf = pre_img_batch(remain_imgs,ends)
        batch_images_tf = torch.Tensor(batch_images_tf)
        batch_images_tf  = batch_images_tf.cuda(0)
        pred_mask = model_t(batch_images_tf)
        pred_mask = pred_mask.cpu()
        pred_mask = pred_mask.detach().numpy() 
        pred_mask1 = np.argmax(pred_mask,axis=1)
    
        for k in range(ends):                      
            pred_mask2 = pred_mask1[k]
            if pred_mask2 == 1:
                benign_id.append(N1+k)
            if pred_mask2 == 2:
                malignant_id.append(N1+k)
       
    end_time = time.time()
    processing_time = end_time - start_time
    #print('Total processing time:',end_time - start_time)
    return benign_id,malignant_id,processing_time
        

##################################################################################################################
##################################################################################################################
 

slide = openslide.open_slide(PATH)
thumbnail = slide.get_thumbnail((slide.dimensions[0]/40, slide.dimensions[1]/40))
plt.figure()
plt.imshow(thumbnail)

all_tissue_samples, binary = find_tissue_patches(PATH)

binary = ~binary
benign_id, malignant_id, processing_time = infer(all_tissue_samples,slide)

h2,w2 = np.shape(binary)
h,w,_ = np.shape(thumbnail)

mask_thumb_all = np.zeros((h2, w2))


for i in range(len(benign_id)):
    a,b=all_tissue_samples.iat[benign_id[i],1]
    mask_thumb_all[a,b] = 1
    

for i in range(len(malignant_id)):
    a,b=all_tissue_samples.iat[malignant_id[i],1]
    mask_thumb_all[a,b] = 2



mask_thumb_all2 = cv2.resize(mask_thumb_all,(w,h),interpolation=cv2.INTER_NEAREST) 
mask_thumb_all3 = bm_color_encode(mask_thumb_all2)
masked_all = np.ma.masked_where(mask_thumb_all3 == 0, mask_thumb_all3)

thumbnail = np.array(thumbnail)
fig, ax = plt.subplots()
ax.imshow(thumbnail)
ax.imshow(masked_all,alpha=0.4)
plt.axis('off')
ax.set_title('benign or malignant')

plt.figure()
plt.imshow(mask_thumb_all3)
plt.title("tile results map")

