# SRE
Implementing Scale Rate Encoding in Darknet & Swin transformer.  

Dataset: Construct a synthetic dataset tailored to specific pathology applications. The dataset should simulate relevant medical imaging conditions to effectively train and evaluate the model.

Original Dataset: The original dataset is not applicable, necessitating the creation of a synthetic alternative.

Pretrained Weights for Backbone: 

  1. Official Darknet GitHub Repository:
  Visit the Darknet GitHub page. Look for links to pretrained weights in the README file.

  2. YOLO Weights Download Page:
  Access the YOLO Weights Page for various versions of YOLO, which utilize Darknet.

>3.Official Swin transformer GitHub Repository:
    Visit the Swin transformer GitHub page. Look for links to pretrained weights in the README file.

  4. Baidu Netdisk: 
  https://pan.baidu.com/s/1D3u-itxpMh9cuo86NchTXg Extracted code: k9jc

Run train_resnet_gmp.py to benchmark the ResNet/DenseNet architecture enhanced with Global Maximum Pooling, including:       
  resnet152 + GMP: network_res_max.py  
  densenet161 + GMP: denseNet_max.py  

# Scale Rate Encoding Implementation in Darknet & Swin Transformer

## Overview
This project implements **Scale Rate Encoding** using two backbone architectures:
- **Darknet** (used in YOLO series)
- **Swin Transformer**

A synthetic dataset is constructed for pathology imaging applications to train and evaluate the models effectively.

---

## Dataset

### Synthetic Dataset Construction
- **Purpose**: Simulate realistic medical imaging conditions in pathology.
- **Features**:
  - Varied staining, noise, and resolution levels.
  - Augmented with scale variations to test scale-encoding robustness.
  - Annotated patches for classification or detection tasks.
- **Rationale**: No suitable public original dataset available; synthetic data ensures controlled evaluation.

---

## Pretrained Weights for Backbone Networks

### 1. Darknet (YOLO)
- **Official GitHub Repository**:  
  [https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)
- **YOLO Pretrained Weights**:  
  [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)
  - Includes weights for YOLOv3, YOLOv4, etc., trained on ImageNet and COCO.

### 2. Swin Transformer
- **Official GitHub Repository**:  
  [https://github.com/microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- **Pretrained Models**:
  - Check the `README.md` for links to models trained on ImageNet.
  - Supports variants: Swin-Tiny, Swin-Small, Swin-Base, Swin-Large.

### 3. Baidu Netdisk (China-friendly access)
- **Link**: [https://pan.baidu.com/s/1D3u-itxpMh9cuo86NchTXg](https://pan.baidu.com/s/1D3u-itxpMh9cuo86NchTXg)
- **Extraction Code**: `k9jc`

---

## Benchmarking ResNet/DenseNet with Global Maximum Pooling (GMP)

Run the following script to benchmark CNN architectures enhanced with **Global Maximum Pooling**:

```bash
python train_resnet_gmp.py
  
     

