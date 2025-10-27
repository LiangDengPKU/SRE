#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dengl
"""

import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import measure
import tensorflow as tf
print(tf.__version__)
process_str_id = str(os.getpid())
import openslide
from skimage.filters import threshold_otsu
from openslide.deepzoom import DeepZoomGenerator
import pandas as pd
import glob
import time
from scipy.ndimage import gaussian_filter

PATH='/'
all_path = glob.glob(PATH + '*.svs')

def remove_extension(path):
    filename_with_ext = os.path.basename(path)
    filename, ext = os.path.splitext(filename_with_ext)
    return filename

os.makedirs('/results',exist_ok=True)
savepath = '/results/'

WIDTH = 256
HEIGHT = 256
N_CLASSES = 3
BATCHSIZE = 8

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

####################################################################################
####################################################################################

def pre_img_batch(imgs_,n_):    
    batch_images_tf_ = imgs_
    batch_images_tf_ = np.array(batch_images_tf_).reshape((n_, WIDTH, HEIGHT, 3))
    batch_images_tf_ = linear_normalize(batch_images_tf_) 
    batch_images_tf_ = np.array(batch_images_tf_,dtype=np.float32)
    return batch_images_tf_

model_name = '/logs/51700res_telang.h5'
model_t = tf.keras.models.load_model(model_name)

def get_rid_off_fp(mask_,thres_,label_,h2_,w2_):    
    all_label, num= measure.label(mask_, return_num=True, connectivity=2) 
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

def neg_pos(all_tissue_samples_,mask1_,mask2_):
    a = np.where(mask1_)
    na = np.shape(a[0])[0]
    b = np.where(mask2_)
    nb = np.shape(b[0])[0]
    nc = len(all_tissue_samples_)
    ne = max(na,nb)
    rate = ne/nc
    print("na,nb",na,nb)
    if rate < 0.016:
        rating="negative slide!"
    else:
        rating="Positive slide!"
    rate2 = 1-rate
    return rate2,rating
"""
pseudo color for speed
"""

def malignant_encode(mask_,w_,h_):
    a = np.where(mask_)    
    a1 = np.shape(a[0])[0]
    for i in range(a1):
        ix2 = np.random.choice(np.arange(2,20), 1)
        mask_[a[0][i],a[1][i]]=ix2[0]
    mask_ = gaussian_filter(mask_,1)
    mask_thumb2 = cv2.resize(mask_,(w_,h_),interpolation=cv2.INTER_CUBIC) 
    mask_thumb2[mask_thumb2 < 1] = 0    
    return mask_thumb2

def benign_encode(mask_,w_,h_):
    a = np.where(mask_)
    a1 = np.shape(a[0])[0]
    for i in range(a1):
        ix2 = np.random.choice(np.arange(2,20), 1)
        mask_[a[0][i],a[1][i]]=ix2[0]

    mask_ = gaussian_filter(mask_,1)
    mask_thumb2 = cv2.resize(mask_,(w_,h_),interpolation=cv2.INTER_CUBIC) 
    mask_thumb2[mask_thumb2 < 1] = 0    
    return mask_thumb2
    
def bn_malg(benign_id_,malignant_id_):
    a = np.where(benign_id_)
    na = np.shape(a[0])[0]
    b = np.where(malignant_id_)
    nb = np.shape(b[0])[0]   
    if nb+na==0:
        precision = 'NA'
        bm_rating = "NA"
    elif nb > na:
        precision = nb/(na+nb)
        bm_rating="malignant!"
    else:
        precision = na/(na+nb)
        bm_rating = "benign!"
    return precision,bm_rating

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
        pred_mask = model_t.predict(batch_images_tf,verbose=0)
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
        pred_mask = model_t.predict(batch_images_tf,verbose=0)
        pred_mask1 = np.argmax(pred_mask,axis=1)
        
        for k in range(ends):                       
            pred_mask2 = pred_mask1[k]
            if pred_mask2 == 1:
                benign_id.append(N1+k)
            if pred_mask2 == 2:
                malignant_id.append(N1+k)    
    end_time = time.time()
    processing_time = end_time - start_time
    return benign_id,malignant_id,processing_time

##################################################################################################################
##################################################################################################################
 
list_file = open(savepath+process_str_id+'test_result.txt', 'a', encoding='utf-8')    
list_file.write('test files path=%s'%(PATH))
list_file.write('\n')

for i in range(len(all_path)):
    name = remove_extension(all_path[i])
    file = all_path[i]    
    slide = openslide.open_slide(file)
    thumbnail = slide.get_thumbnail((slide.dimensions[0]/40, slide.dimensions[1]/40))    
    all_tissue_samples, binary = find_tissue_patches(file)    
    binary = ~binary   
    benign_id, malignant_id, processing_time = infer(all_tissue_samples,slide)   
    h2,w2 = np.shape(binary)
    h,w,_ = np.shape(thumbnail)    
    mask_thumb_all = np.zeros((h2, w2))    
    thres = 20    
    for i in range(len(benign_id)):
        a,b=all_tissue_samples.iat[benign_id[i],1]
        mask_thumb_all[a,b] = 1       
    mask1 = get_rid_off_fp(mask_thumb_all,thres,1,h2,w2)   
    mask_thumb_all = np.zeros((h2, w2))   
    for i in range(len(malignant_id)):
        a,b=all_tissue_samples.iat[malignant_id[i],1]
        mask_thumb_all[a,b] = 2    
    mask2 = get_rid_off_fp(mask_thumb_all,thres,2,h2,w2)   
    mask_thumb_all = np.zeros((h2, w2))   
    mask_thumb_all = mask_thumb_all + mask1 + mask2   
    rate,rating = neg_pos(all_tissue_samples,mask1,mask2)
    precision,bm_rating = bn_malg(mask1,mask2)    
    mask_thumb_all2 = cv2.resize(mask_thumb_all,(w,h),interpolation=cv2.INTER_NEAREST)     
    mask_thumb_all3 = bm_color_encode(mask_thumb_all2)   
    masked_all = np.ma.masked_where(mask_thumb_all3 == 0, mask_thumb_all3)    
    thumbnail = np.array(thumbnail)
     
    if len(malignant_id) > len(benign_id):
        mask_thumb2 = malignant_encode(mask2,w,h)
    else:
        mask_thumb2 = benign_encode(mask1,w,h)  
    masked = np.ma.masked_where(mask_thumb2 == 0, mask_thumb2)
   
    ##############################################################################################
    "savefig"
    ##############################################################################################        
    savepath_1 = savepath+name
    os.makedirs(savepath_1,exist_ok=True)

    
    fig, ax = plt.subplots()
    ax.imshow(thumbnail)
    ax.imshow(masked_all,alpha=0.4)
    plt.axis('off')
    ax.set_title('benign or malignant')
    plt.savefig(savepath_1 + '/'+'vote.jpg')
   
    fig, ax = plt.subplots()
    ax.imshow(thumbnail)
    ax.imshow(masked,'jet',alpha=0.7)
    ax.set_title('heatmap')
    plt.axis('off')
    plt.savefig(savepath_1 + "/" + 'heatmap.jpg')
       
    fig, ax = plt.subplots()
    ax.imshow(thumbnail)
    plt.axis('off')
    ax.set_title('slide')
    plt.savefig(savepath_1 + "/"+'slide.jpg')
    
    plt.close('all')
    
    ##############################################################################################
    "savetxt"
    ##############################################################################################        
    list_file = open(savepath+process_str_id+'test_result.txt', 'a', encoding='utf-8')  
    list_file.write('slide name=%s'%(name))
    list_file.write('\n')   
    list_file.write('processing time=%s'%(str(processing_time)))
    list_file.write('\n')        
    list_file.write('positive or negative=%s,rating=%s'%(rating,str(rate)))
    list_file.write('\n')   
    list_file.write('benign or malignant=%s,rating=%s'%(bm_rating,str(precision)))
    list_file.write('\n')
    list_file.write('\n')      
    list_file.close()    

