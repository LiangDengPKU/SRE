"""
@author: Liang Deng
"""
import time    
import datetime
import os
os.chdir('/media/mediway/Work2/deep3/SRE-master2')
import numpy as np
import torch
print(torch.__version__)
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from SREswin import SreSwin
#from torch.nn import CrossEntropyLoss
from nets.yolo_training import (ModelEMA, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import LossHistory
from utils.utils_fit import fit_one_epoch
from pathology_dataloader import data_gen_jpg_sre, val_gen_jpg, val_train_split, contruct_jpg
import glob
import torch.nn.functional as F
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
path_benign_rim_small = '/media/mediway/Work2/dataset7/telang_small/1/'
path_ma_rim_small     = '/media/mediway/Work2/dataset7/telang_small/2/'
path_hemo_small       = '/media/mediway/Work2/dataset7/telang_small/3/'
path_normalb_small    = '/media/mediway/Work2/dataset7/telang_small/4/'
path_normalt_small    = '/media/mediway/Work2/dataset7/telang_small/5/'
path_cartilage_small  = '/media/mediway/Work2/dataset7/telang_small/6/'

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



class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss for multi-class classification.
        Args:
            alpha (list or float, optional): Class weights. If None, no weighting.
            gamma (float): Focusing parameter. Default: 2.0.
            reduction (str): 'none', 'mean', or 'sum'. Default: 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (N, C) logits
        targets: (N,) class indices
        """
        # Compute log probabilities
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        # Gather probabilities for true classes
        true_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
        log_true_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - true_probs) ** self.gamma

        # Apply alpha (class weights)
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
            else:
                alpha_t = self.alpha
            focal_weight = alpha_t * focal_weight

        # Focal loss
        loss = -focal_weight * log_true_probs

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


if __name__ == "__main__":
    Cuda            = True
    distributed     = False
    sync_bn         = False
    fp16            = False

    #------------------------------------------------------#  
    #input_shape
    #------------------------------------------------------#
    ORIGINIAL_DIMENSION = 256
    input_shape = [640,640]
    one_cell_fov=64
    #------------------------------------------------------#
    #------------------------------------------------------#
    #------------------------------------------------------#
    pretrained      = True
    #------------------------------------------------------#
    #------------------------------------------------------#
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
    model = SreSwin(num_classes=num_classes,pretrained=pretrained,input_shape=input_shape)
    
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
    
    criterion = FocalLoss(alpha=[0.3, 0.5, 0.7], gamma=2.0)

    num_train   = 1000
    num_val     = 100


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
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                # ... 解冻 backbone、调整 batch_size、调整学习率 ...
                UnFreeze_flag = True
        
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        
            # 每个 epoch 随机生成 scale_rate
            np.random.seed(epoch + int(time.time()))
            scale_rate = np.random.choice(np.arange(one_cell_fov, ORIGINIAL_DIMENSION), 1)[0]
            print(f"Epoch {epoch}: scale_rate = {scale_rate}")

            train_gen = data_gen_jpg_sre(train_benign_rim_jpg,train_ma_rim_jpg,train_hemo_jpg,train_bg_jpg,scale_rate,batch_size)
            val_gen = val_gen_jpg(val_benign_rim_jpg,val_ma_rim_jpg,val_hemo_jpg,val_bg_jpg,batch_size)
        
            # 传入 scale_rate 到训练函数
            fit_one_epoch(
                model_train, model, ema, criterion, loss_history, optimizer,
                epoch, epoch_step, epoch_step_val, train_gen, val_gen,
                UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank,
                scale_rate=scale_rate
            )
            
            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()