#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:59:50 2024

@author: mediway
"""



scale_rate = 200
train_gen = data_gen_jpg_sre(train_benign_rim_jpg,train_ma_rim_jpg,train_hemo_jpg,train_bg_jpg,scale_rate)
val_gen = val_gen_jpg(val_benign_rim_jpg,val_ma_rim_jpg,val_hemo_jpg,val_bg_jpg)