# Scale Rate Encoding Implementation in Darknet & Swin Transformer

## Overview
This project implements **Scale Rate Encoding** using two backbone architectures:
- **Darknet** (used in YOLO series)
- **Swin Transformer**

---

## Dataset

### Synthetic Dataset Construction
- **Purpose**: Simulate realistic medical imaging conditions in pathology.
- **Features**:
  - Varied staining, noise, and resolution levels.
  - Augmented with scale variations to test scale-encoding robustness.
  - Annotated patches for classification or detection tasks.
- **Original Dataset**: The original dataset is not applicable, necessitating the creation of a synthetic alternative.

---

## Pretrained Weights for Backbone Networks

### 1. Darknet (YOLO)
- **Official GitHub Repository**:  
- **YOLO Pretrained Weights**:  
  - Includes weights for YOLOv3, YOLOv4, etc., trained on ImageNet and COCO.

### 2. Swin Transformer
- **Official GitHub Repository**:  
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
train_resnet_gmp.py
resnet152 + GMP: network_res_max.py
densenet161 + GMP: denseNet_max.py 

  
     

