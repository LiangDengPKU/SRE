import os
import torch
from tqdm import tqdm
from utils.utils import get_lr
        
def fit_one_epoch(model_train, model, ema, criterion, loss_history, optimizer, epoch, epoch_step, epoch_step_val, train_gen, val_gen, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss        = 0
    val_loss    = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3,position=0)
    model_train.train()
    for iteration, batch in enumerate(train_gen):
        if iteration >= epoch_step:
            break

        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = torch.Tensor(images)

                images  = images.cuda(local_rank)


        device = torch.device('cuda:0')        

        ground_truth = torch.Tensor(targets).to(device)
        ground_truth = ground_truth.long().to(device)
        
        optimizer.zero_grad()

        #----------------------#
        #   forward
        #----------------------#
        outputs         = model_train(images)

        loss_value = criterion(outputs, ground_truth)

        #----------------------#
        #   backward
        #----------------------#
        loss_value.backward()
        optimizer.step()

        if ema:
            ema.update(model_train)

        loss += loss_value.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3,position=0)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
        
    for iteration, batch in enumerate(val_gen):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = torch.Tensor(images)
                images  = images.cuda(local_rank)

            #----------------------#
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   forward
            #----------------------#
            outputs         = model_train_eval(images)
            device = torch.device('cuda:0')        

            ground_truth = torch.Tensor(targets).to(device)
            ground_truth = ground_truth.long().to(device)

            loss_value = criterion(outputs, ground_truth)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   save
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))