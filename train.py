"""
@author: Liang Deng
"""
    
import datetime
import os
os.chdir('/SRE-master')

import numpy as np
import torch
print(torch.__version__)
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from SREdarknet import DarkNet
from torch.nn import CrossEntropyLoss
from nets.yolo_training import (ModelEMA, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import LossHistory
from utils.utils_fit import fit_one_epoch
from pathology_dataloader import data_gen_jpg_sre, val_gen_jpg, val_train_split, contruct_jpg
import glob

"""
Identification of Tumor Cell Clusters in Histopathology Images Using a Deep Learning Model with Scale Rate Encoding
specifically telangiectatic osteosarcoma (malignant) and aneurysmal bone cysts (benign).

benign_rim: non-malignant tumor cell clusters
ma_rim:     malignant tumor cell clusters
hemo:       blood-filled cystic spaces
normalb:    normal bone
normalt:    normal tissue
cartilage:  cartilage

"""
path_benign_rim_small = '/dataset/benign_rim/'
path_ma_rim_small     = '/dataset/ma_rim/'
path_hemo_small       = '/dataset/hemo/'
path_normalb_small    = '/dataset/normalb/'
path_normalt_small    = '/dataset/normalt/'
path_cartilage_small  = '/dataset/cartilage/'



benign_rim_path = glob.glob(path_benign_rim_small + '*.jpeg')
ma_rim_path     = glob.glob(path_ma_rim_small + '*.jpeg')
hemo_path       = glob.glob(path_hemo_small + '*.jpeg')
normalb_path    = glob.glob(path_normalb_small + '*.jpeg')
normalt_path    = glob.glob(path_normalt_small + '*.jpeg')
cartilage_path  = glob.glob(path_cartilage_small + '*.jpeg')

benign_rim_path.sort()
ma_rim_path.sort()
hemo_path.sort()
normalb_path.sort()
normalt_path.sort()
cartilage_path.sort()

train_benign_rim, val_benign_rim = val_train_split(benign_rim_path)
train_ma_rim, val_ma_rim         = val_train_split(ma_rim_path)
train_hemo, val_hemo             = val_train_split(hemo_path)
train_normalb, val_normalb       = val_train_split(normalb_path)
train_normalt, val_normalt       = val_train_split(normalt_path)
train_cartilage, val_cartilage   = val_train_split(cartilage_path)


train_bg = train_normalb + train_normalt + train_cartilage
val_bg   = val_normalb + val_normalt + val_cartilage
print("loading train_benign_rim")
train_benign_rim_jpg = contruct_jpg(train_benign_rim)
print("loading train_ma_rim")
train_ma_rim_jpg     = contruct_jpg(train_ma_rim)
print("loading train_hemo")
train_hemo_jpg       = contruct_jpg(train_hemo)
print("loading train_bg")
train_bg_jpg         = contruct_jpg(train_bg)

print("loading val_benign_rim")
val_benign_rim_jpg = contruct_jpg(val_benign_rim)
print("loading val_ma_rim")
val_ma_rim_jpg     = contruct_jpg(val_ma_rim)
print("loading val_hemo")
val_hemo_jpg       = contruct_jpg(val_hemo)
print("loading val_bg")
val_bg_jpg         = contruct_jpg(val_bg)

if __name__ == "__main__":
    #---------------------------------#
    #---------------------------------#
    Cuda            = True
    #---------------------------------#
    #---------------------------------#

    #---------------------------------#
    #   DP：
    #       distributed = False
    #       CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP：
    #       distributed = True
    #       CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     DDP mode
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        > pytorch1.7.1
    #---------------------------------------------------------------------#
    fp16            = False
    #----------------------------------------------------------------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = 'model_data/yolov5_s.pth'
    #------------------------------------------------------#
    #   input_shape     
    #------------------------------------------------------#
    input_shape     = [256, 256]
    #------------------------------------------------------#

    #------------------------------------------------------#
    backbone        = 'cspdarknet'
    #------------------------------------------------------#
    #------------------------------------------------------#
    pretrained      = True
    #------------------------------------------------------#
    #   phi             s、m、l、x
    #------------------------------------------------------#
    phi             = 's'
    #------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------------------------#
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 300，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4
    #           Init_Epoch = 0，UnFreeze_Epoch = 300，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 16
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 300
    Unfreeze_batch_size = 8
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    Freeze_Train        = True
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  
    #                   Adam  Init_lr=1e-3
    #                   SGD   Init_lr=1e-2
    #   weight_decay    
    #                   adam  0 
    #                   SGD   5e-4
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4
    #------------------------------------------------------------------#
    #   lr_decay_type   step、cos
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    save_period         = 10
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------#
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    #------------------------------------------------------#
    #------------------------------------------------------#
    num_classes = 3

    #------------------------------------------------------#
    #------------------------------------------------------#
    model = DarkNet(num_classes, phi)
    
    if not pretrained:
        weights_init(model)

    #----------------------#
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
            
    ema = ModelEMA(model_train)
    
    criterion = CrossEntropyLoss()

    num_train   = 100000
    num_val     = 2000

    if local_rank == 0:

        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('Not enough data')

    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        #-------------------------------------------------------------------#
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #---------------------------------------#
        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   
        optimizer = {
            'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        #---------------------------------------#
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if ema:
            ema.updates     = epoch_step * Init_Epoch
           
        #---------------------------------------#
        #   start training
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   learning rate setting
                #-------------------------------------------------------------------#
                nbs             = 64
                lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Not enough data")
                    
                if ema:
                    ema.updates     = epoch_step * epoch

                if distributed:
                    batch_size  = batch_size // ngpus_per_node

                UnFreeze_flag   = True

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            scale_rate = np.random.choice(np.arange(64,256),1)[0] 
            model.scale_rate = scale_rate
            train_gen = data_gen_jpg_sre(train_benign_rim_jpg,train_ma_rim_jpg,train_hemo_jpg,train_bg_jpg,scale_rate,batch_size)
            val_gen = val_gen_jpg(val_benign_rim_jpg,val_ma_rim_jpg,val_hemo_jpg,val_bg_jpg,batch_size)
            print("scale_rate:",model.scale_rate)
            print("batchsize:",batch_size)

            fit_one_epoch(model_train, model, ema, criterion, loss_history, optimizer, epoch, epoch_step, epoch_step_val, train_gen, val_gen, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)
            
            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()