# based on https://github.com/aschnelle-dev/equivariant-SR-turbulent-flows

from tqdm.auto import tqdm

import math
import torch
import numpy as np

from torch.utils.data import DataLoader

import os
import json
import glob
import re
import time

from copy import deepcopy
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from .data_augmentation import DataAugmentation


def train(model: torch.nn.Module, 
          models_dir: str, 
          model_name: str, 
          train_name: str,
          start_epoch: int, 
          epochs: int, 
          train_loader: DataLoader, 
          valid_loader: DataLoader, 
          loss_fn, 
          optimizer, 
          lr_scheduler, 
          use_lr_scheduler: bool, 
          early_stopping: int, 
          only_save_best: bool, 
          train_samples: int, 
          batch_size: int,
          data_augmentation: DataAugmentation, 
          plot: bool,
          accumulated_batches: int = 1,
          early_stopping_threshold: float = 1e-5,
          initial_early_stop_count: int = 0,
          train_loss_in_eval: bool = False,
          model_forward_kwargs: dict = {}):
    
    tb_dir = os.path.join(os.path.dirname(os.path.abspath(models_dir)), 'runs')
    writer = SummaryWriter(os.path.join(tb_dir, model_name, train_name)) # Tensorboard writer

    output_dir = os.path.join(models_dir, model_name, train_name)
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'log.json')

    if start_epoch == 0:
        best_loss = np.inf
        best_epoch = -1
        train_loss_values = []
        valid_loss_values = []
        epoch_duration_values = []
        lr_values = []
    else:
        best_loss = compute_loss(valid_loader, model, loss_fn, model_forward_kwargs)
        best_epoch = start_epoch
        with open(log_file, 'r') as f:
            log_dict = json.load(f)
            train_loss_values = log_dict['train_loss'][:start_epoch]
            valid_loss_values = log_dict['valid_loss'][:start_epoch]
            epoch_duration_values = log_dict['epoch_duration'][:start_epoch]
            lr_values = log_dict['lr'][:start_epoch]
        
    early_stop_count = initial_early_stop_count
    
    with tqdm(initial=start_epoch, total=start_epoch+epochs, desc='training', unit='epoch', dynamic_ncols=True) as pbar:
        for epoch in range(1+start_epoch, 1+start_epoch+epochs):
            epoch_start = time.time()
            train_loss, valid_loss = train_loop(train_loader, valid_loader, model, loss_fn, optimizer, 
                                                epoch, batch_size, train_samples, accumulated_batches,
                                                data_augmentation, train_loss_in_eval, model_forward_kwargs)
            epoch_duration = time.time() - epoch_start
            pbar.update(1)
            
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/valid', valid_loss, epoch)
            train_loss_values.append(train_loss)
            valid_loss_values.append(valid_loss)
            epoch_duration_values.append(epoch_duration)
            
            if use_lr_scheduler:
                lr_scheduler.step(valid_loss)
            lr_values.append(lr_scheduler.get_last_lr()[0])
                
            if valid_loss < best_loss - early_stopping_threshold:
                early_stop_count = 0
                
                best_loss = valid_loss
                best_epoch = epoch
                best_weights = deepcopy(model.state_dict())
                best_optim = optimizer.state_dict()
                pbar.set_postfix_str(f'{best_epoch=:}, {train_loss=:.5f}, {valid_loss=:.5f}')
                
                if only_save_best:
                    remove_saved_models(output_dir)
                
                output_file = os.path.join(output_dir, f'epoch{epoch}.tar')
                save_checkpoint(output_file, best_weights, best_optim, early_stop_count)
            else:
                early_stop_count += 1
                
            save_log(log_file, train_loss_values, valid_loss_values, epoch_duration_values,
                     best_epoch, best_loss, lr_values)
            
            if early_stop_count >= early_stopping:
                print(f'Early stopping at epoch {epoch}.')
                break
        else:
            print(f'Finished training after {epoch} epochs without early stopping.')
        
        print(f'Best epoch is epoch {best_epoch} with a validation loss of {best_loss}.')
        print(f'Trained model is saved in {output_dir}')
        print(f'A log of both training and validation loss is saved in {log_file}')
            
    writer.flush()
    writer.close()
    
    if plot:
        epochs = range(1, start_epoch+epochs+1)
        plt.plot(epochs, train_loss_values, label="train")
        plt.plot(epochs, valid_loss_values, label="valid")
        plt.xticks(epochs)
        
        plt.title(f"{model_name} ({train_name})  Best loss: {best_loss:>.3f}   Best epoch: {best_epoch}")
        plt.legend()
        
        plt.show()
    

def train_loop(train_loader: DataLoader, 
               valid_loader: DataLoader, 
               model: torch.nn.Module,
               loss_fn, 
               optimizer, 
               epochnum: int, 
               batch_size: int,
               samples: int,
               accumulated_batches: int = 1,
               data_augmentation: DataAugmentation = None, 
               train_loss_in_eval: bool = False,
               model_forward_kwargs: dict = {}):
    model.train()
    running_loss = 0.0
    
    n = 0
    with tqdm(total=math.ceil(samples/batch_size), desc=f'epoch {epochnum}', unit='batch', dynamic_ncols=True) as pbar:
        optimizer.zero_grad() # Resets gradient
        for i, (x, y) in enumerate(train_loader, 1):           
            if data_augmentation:
                x, y = data_augmentation(x, y)
                
            train_model_forward_kwargs = model_forward_kwargs.copy()
            if 'epoch' in train_model_forward_kwargs:
                train_model_forward_kwargs['epoch'] = epochnum
            if 'ground_truth' in train_model_forward_kwargs:
                train_model_forward_kwargs['ground_truth'] = y

            # Compute prediction and loss
            pred = model(x, **train_model_forward_kwargs)

            loss = loss_fn(pred, y)
            cur_batch_size = x.size(0)
            running_loss += loss.item()*cur_batch_size
        
            # accumulate gradient updates over multiple batches
            (loss/accumulated_batches).backward()
            
            if i % accumulated_batches == 0:
                 # Adjusts the parameters by the gradients accumulated in the backward passes
                optimizer.step()
                optimizer.zero_grad() # Resets gradient
            
            n += cur_batch_size
            pbar.set_postfix_str(f'train_loss={running_loss/n:.5f}')
            pbar.update(1)
        
        # check whether batches remain without weight update    
        if i % accumulated_batches > 0:
            # Adjusts the parameters by the gradients accumulated in the backward passes
            optimizer.step()
            
        if train_loss_in_eval:
            train_loss = compute_loss(train_loader, model, loss_fn, model_forward_kwargs)
        else:
            train_loss = running_loss / n
        valid_loss = compute_loss(valid_loader, model, loss_fn, model_forward_kwargs)
        
        pbar.set_postfix_str(f'{train_loss=:.5f}, {valid_loss=:.5f}')
    
    return train_loss, valid_loss


def compute_loss(dataloader: DataLoader, model: torch.nn.Module, loss_fn, model_forward_kwargs: dict = {}):
    model.eval()
    valid_loss = 0.0
    n = 0
    with torch.no_grad(): # Ensures that no gradients are computed during test mode
        for i, (x, y) in enumerate(dataloader, 1):
            pred = model(x, **model_forward_kwargs)
            batch_size = x.size(0)
            valid_loss += loss_fn(pred, y).item()*batch_size
            n += batch_size
    return valid_loss / n


def save_checkpoint(path, weights, optimizer_state, early_stop_count):
    torch.save({
        'model_state_dict': weights,
        'optimizer_state_dict': optimizer_state,
        'early_stop_count': early_stop_count
    }, path)
    
    
def save_log(log_file, train_loss_values, valid_loss_values, epoch_duration_values, 
             best_epoch, best_loss, lr_values):
    with open(log_file, 'w+') as f:
        log_dict = {'train_loss': train_loss_values, 
                    'valid_loss': valid_loss_values,
                    'epoch_duration': epoch_duration_values,
                    'best_epoch': best_epoch,
                    'best_valid_loss': best_loss,
                    'lr': lr_values}
        json.dump(log_dict, f, indent=4)
    

def load_trained_model(model, models_dir, model_name, train_name, optimizer=None, epoch=-1):
    directory = os.path.join(models_dir, model_name, train_name)
    
    if epoch == 0:
        return 0, 0
    if not os.path.exists(directory) or not os.listdir(directory):
        if epoch == -1: return 0, 0
        raise Exception('no saved model to load')
    elif epoch == -1:
        # load latest epoch
        pattern = re.compile(r"epoch([0-9]+)\.tar")
        saved_epochs = [filename for filename in os.listdir(directory) if pattern.fullmatch(filename)]
        last_epoch_file = max(saved_epochs, key=lambda f: int(pattern.fullmatch(f).group(1)), default=None)
        if last_epoch_file is None:
            return 0, 0
        epoch = int(pattern.fullmatch(last_epoch_file).group(1))
        file_path = os.path.join(directory, last_epoch_file)
    else:
        # load specified epoch
        file_path = os.path.join(directory, f'epoch{epoch}.tar')
        if not os.path.isfile(file_path):
            raise Exception('model not saved for that epoch')
    
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if optimizer: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    early_stop_count = checkpoint['early_stop_count']
    model.eval()
    
    print(f"Loaded state at epoch {epoch} with an early stop count of {early_stop_count}.")
    
    return early_stop_count, epoch
    
    
def remove_saved_models(directory):
    path_pattern = os.path.join(directory, 'epoch*.tar')

    for file_path in glob.glob(path_pattern):
        os.remove(file_path)