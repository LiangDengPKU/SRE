# Scale Rate Encoding Implementation in Darknet & Swin Transformer

## Overview
This project implements **Scale Rate Encoding (SRE)** using two backbone architectures:
- **Darknet** (used in YOLO series)
- **Swin Transformer**

## Usage

### 1. Inference with CSPDarknet + SRE
Run inference on a whole-slide image (.svs) using the CSPDarknet backbone with Scale Rate Encoding.
```bash
python infer_SREDarknet.py
```

### 2. Inference with Swin Transformer + SRE
Run inference on a whole-slide image (.svs) using the Swin Transformer backbone with Scale Rate Encoding.
```bash
python infer_SREswin.py
```

### 3. Inference with ResNet (Baseline)
Run inference on a whole-slide image (.svs) using the standard ResNet backbone for comparison.
```bash
python infer_resnet.py
```


![heatmap](./results/heatmap.jpg)

---

## Dataset
The original dataset is not publicly available. However, the method is implementable for pathology Whole Slide Images (WSIs), including but not limited to osteosarcoma, lymph node metastases, and lung cancer.
*   Whole Slide Images are provided in `.svs` format.
*   These WSIs are sliced into smaller patches in `.jpg` format for model training and inference.


## Pretrained Weights (updating)

### 1. For Backbone Networks
- **Link**: https://pan.baidu.com/s/1pEkNFg-Exv5Hv2wraQHIMw?pwd=mnzy 
- **Extraction Code**: `mnzy`

### 2. For quick inference test
- **Link**: https://pan.baidu.com/s/1md_8DW8TC0eGgaNVpj7QBQ?pwd=67fb
- **Extraction Code**: `67fb`



  
     

