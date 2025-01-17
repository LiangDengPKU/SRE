
"""
@author: Liang Deng
"""

import cv2
import numpy as np
from skimage import exposure


FOV = 362


def val_train_split(data):
    np.random.seed(77)
    np.random.shuffle(data)
    split_point = int(len(data) * 0.8)
    train_data = data[:split_point]
    val_data = data[split_point:]
    return train_data,val_data

def contruct_jpg(pathlist):
    features_jpg = []
    for i in range(len(pathlist)):
        temp = cv2.imread(pathlist[i])
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        features_jpg.append(temp)
    return features_jpg

def linear_normalize(img):
    img_ln = img/255.0
    return img_ln    
 
def minmax_normalize(img):
    img_mn = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img_mn 
  
def hsv_trans(image,hue=.1, sat=0.7, val=0.4):
    r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    image_data      = np.array(image, np.uint8)
    hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))

    dtype = image_data.dtype
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
    return image_data    

def jitter(img,step):
    
    i_bright = np.random.randint(1,7)
    img_temp = img
                    
    if i_bright == 1:
        gamma_i = np.random.choice(np.arange(0.5,1.55,step), 1)
        img_temp = exposure.adjust_gamma(img_temp, gamma_i[0])

    elif i_bright == 2:
        img_temp=hsv_trans(img_temp,hue=.1, sat=0.7, val=0.4)

    elif i_bright == 3:
        img_temp = exposure.adjust_log(img_temp, 1)

    elif i_bright == 5:
        p2, p98 = np.percentile(img_temp, (2, 98))
        img_temp = exposure.rescale_intensity(img_temp, in_range=(p2, p98)) 
    else:
        img_temp = img_temp
    return img_temp


def rotate_aug(img_,height,dense):
    #print("rotation")
    angles = []
    angles_dense = []
    
    angle_1 = 0    
    for i in range(360):
        angle_1 = angle_1 + 1
        angles.append(angle_1)
    angle_2 = 0        
    for i in range(3600):
        angle_2 = angle_2 + 0.1
        angles_dense.append(angle_2)     

    angle2 = np.random.choice(angles)            
    mat = cv2.getRotationMatrix2D((FOV/2,FOV/2),angle2,1)
    
    angle2_dense = np.random.choice(angles_dense)            
    mat_dense = cv2.getRotationMatrix2D((FOV/2,FOV/2),angle2_dense,1)
    
    if dense:
        img_1 = cv2.warpAffine(img_,mat_dense,(FOV,FOV),flags=cv2.INTER_NEAREST)
        #print("dense")
    else:
        img_1 = cv2.warpAffine(img_,mat,(FOV,FOV),flags=cv2.INTER_NEAREST)
        #print("not dense")
    img_2 = img_1[int(FOV/2-height/2):int(FOV/2+height/2),int(FOV/2-height/2):int(FOV/2+height/2)]
    return img_2

def translate_aug(img_,height):
    #print("translation")
    dx = FOV - height
    dx_1 = np.random.choice(range(dx))
    dy_1 = np.random.choice(range(dx))
    img_2 = img_[dx_1:(dx_1+height),dy_1:(dy_1+height)]
    return img_2
    

"""
normal tissue: 0
benign: 1
malignant: 2
hemo: 0
normal bone: 0

"""


def data_gen_jpg_sre(train_benign_rim_jpg,train_ma_rim_jpg,train_hemo_jpg,train_bg_jpg,scale_rate,batchsize):
    
    while True:
        """
        scale_rate:64~256
        """        
        gforce = int(batchsize/8)

        N_benign = 2*gforce
        N_malignant = 3*gforce
        N_hemo = 1*gforce
        N_bg = 2*gforce
        
        imgs = []
        labels = []
        
        rot_trans = [0,1]
        flips = [0,1,-1,2]         
        
        height = scale_rate

        """
        benign
        """        
        N = len(train_benign_rim_jpg)         
        ix = np.random.choice(np.arange(N), N_benign, replace=False)
        for i in range(N_benign):
            #print(i)
            img_1 = train_benign_rim_jpg[ix[i]]
             
            flip2 = np.random.choice(flips)      
            if flip2 != 2:
                img_1 = cv2.flip(img_1,flip2)                

            pp = np.random.choice(rot_trans)            
            if pp == 0:
                img_2 = rotate_aug(img_1,height,dense=True)
            else:
                img_2 = translate_aug(img_1,height)
                      
            img_2 = jitter(img_2,0.01)
            
            img_2 = cv2.resize(img_2,(640,640),interpolation=cv2.INTER_CUBIC)
            img_2 = linear_normalize(img_2)
            img_2 = np.transpose(img_2,(2,0,1))
            imgs.append(img_2)
            labels.append(1)

        """
        malignant
        """

        N2 = len(train_ma_rim_jpg)         
        ix2 = np.random.choice(np.arange(N2), N_malignant, replace=False)
            
        for i in range(N_malignant):
            img_1 = train_ma_rim_jpg[ix2[i]]

            flip2 = np.random.choice(flips)  
            if flip2 != 2:
                img_1 = cv2.flip(img_1,flip2)

            pp = np.random.choice(rot_trans)           
            if pp == 0:
                img_2 = rotate_aug(img_1,height,dense=False)
            else:
                img_2 = translate_aug(img_1,height)
            img_2 = jitter(img_2,0.05)  
            img_2 = cv2.resize(img_2,(640,640),interpolation=cv2.INTER_CUBIC)
            img_2 = linear_normalize(img_2)
            img_2 = np.transpose(img_2,(2,0,1))
            imgs.append(img_2)
            labels.append(2)


        """
        hemo
        """
        N3 = len(train_hemo_jpg)         
        ix2 = np.random.choice(np.arange(N3), N_hemo, replace=False)

        for i in range(N_hemo):

            img_1 = train_hemo_jpg[ix2[i]]

            flip2 = np.random.choice(flips)  
            if flip2 != 2:
                img_1 = cv2.flip(img_1,flip2)

            
            pp = np.random.choice(rot_trans)            
            if pp == 0:
                img_2 = rotate_aug(img_1,height,dense=True)
            else:
                img_2 = translate_aug(img_1,height)

            img_2 = jitter(img_2,0.01)
            img_2 = cv2.resize(img_2,(640,640),interpolation=cv2.INTER_CUBIC)
            img_2 = linear_normalize(img_2)   
            img_2 = np.transpose(img_2,(2,0,1))
            imgs.append(img_2)
            labels.append(0)

        """
        others
        """
        N4 = len(train_bg_jpg)         
        ix2 = np.random.choice(np.arange(N4), N_bg,replace=False)

        for i in range(N_bg):

            img_1 = train_bg_jpg[ix2[i]]

            flip2 = np.random.choice(flips) 
            if flip2 != 2:
                img_1 = cv2.flip(img_1,flip2)

            pp = np.random.choice(rot_trans)            
            if pp == 0:
                img_2 = rotate_aug(img_1,height,dense=False)
            else:
                img_2 = translate_aug(img_1,height)

            img_2 = jitter(img_2,0.05)
            img_2 = cv2.resize(img_2,(640,640),interpolation=cv2.INTER_CUBIC)
            img_2 = linear_normalize(img_2)
            img_2 = np.transpose(img_2,(2,0,1))

            imgs.append(img_2)
            labels.append(0)

        state = np.random.get_state()
        np.random.shuffle(imgs)
        np.random.set_state(state)
        np.random.shuffle(labels)
        imgs = np.array(imgs).reshape((batchsize, 3, 640, 640))
        imgs = np.array(imgs,dtype=np.float32)
        labels = np.array(labels,dtype=np.uint8)    
        
        yield imgs,labels




def val_gen_jpg(val_benign_rim_jpg,val_ma_rim_jpg,val_hemo_jpg,val_bg_jpg,batchsize):
   
    while True:       
        gforce = int(batchsize/8)

        N_benign = 2*gforce
        N_malignant = 3*gforce
        N_hemo = 1*gforce
        N_bg = 2*gforce
        
        imgs = []
        labels = []
        
        """
        benign
        """        
        N = len(val_benign_rim_jpg)         
        ix = np.random.choice(np.arange(N), N_benign, replace=False)
        for i in range(N_benign):
            #print(i)
            img_1 = val_benign_rim_jpg[ix[i]]
            img_2 = translate_aug(img_1,256)
            img_2 = cv2.resize(img_2,(640,640),interpolation=cv2.INTER_CUBIC)
            img_2 = linear_normalize(img_2)
            img_2 = np.transpose(img_2,(2,0,1))
            imgs.append(img_2)
            labels.append(1)

        """
        malignant
        """

        N2 = len(val_ma_rim_jpg)         
        ix2 = np.random.choice(np.arange(N2), N_malignant, replace=False)
            
        for i in range(N_malignant):
            img_1 = val_ma_rim_jpg[ix2[i]]
            img_2 = translate_aug(img_1,256)      
            img_2 = cv2.resize(img_2,(640,640),interpolation=cv2.INTER_CUBIC)
            img_2 = linear_normalize(img_2)
            img_2 = np.transpose(img_2,(2,0,1))
            imgs.append(img_2)
            labels.append(2)

        """
        hemo
        """
        N3 = len(val_hemo_jpg)         
        ix2 = np.random.choice(np.arange(N3), N_hemo, replace=False)

        for i in range(N_hemo):
            img_1 = val_hemo_jpg[ix2[i]]
            img_2 = translate_aug(img_1,256)
            img_2 = cv2.resize(img_2,(640,640),interpolation=cv2.INTER_CUBIC)
            img_2 = linear_normalize(img_2)  
            img_2 = np.transpose(img_2,(2,0,1))            
            imgs.append(img_2)
            labels.append(0)


        """
        others
        """
        N4 = len(val_bg_jpg)         
        ix2 = np.random.choice(np.arange(N4), N_bg, replace=False)

        for i in range(N_bg):
            img_1 = val_bg_jpg[ix2[i]]
            img_2 = translate_aug(img_1,256)
            img_2 = cv2.resize(img_2,(640,640),interpolation=cv2.INTER_CUBIC)
            img_2 = linear_normalize(img_2)       
            img_2 = np.transpose(img_2,(2,0,1))
            imgs.append(img_2)
            labels.append(0)
     
        state = np.random.get_state()
        np.random.shuffle(imgs)
        np.random.set_state(state)
        np.random.shuffle(labels)

        imgs = np.array(imgs).reshape((batchsize, 3, 640, 640))
        imgs = np.array(imgs,dtype=np.float32)
        labels = np.array(labels,dtype=np.uint8)    
        
        yield imgs,labels


