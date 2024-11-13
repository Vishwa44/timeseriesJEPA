from data_provider.data_factory import data_provider
from data_provider.mask_collator import TimeSeriesMaskCollator
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from model.PatchTST_encoder import PatchTST_embedding
from model.PatchTST_predictor import PatchTST_predictor
from model.PatchTST_finetune import PatchTST_finetune
from data_provider.mask_utils import apply_masks

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

import copy
import os
import time

from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import numpy as np

def _get_data(args, flag, collator=None):
        data_set, data_loader = data_provider(args, flag, collator)
        return data_set, data_loader



def vali(args, model, device, vali_data, vali_loader, criterion):
    total_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(vali_loader)):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            # encoder - decoder
            
            if 'Linear' in args.model or 'TST' in args.model:
                outputs = model(batch_x)
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true)

            total_loss.append(loss)
    total_loss = np.average(total_loss)
    model.train()
    return model, total_loss

def finetune(args, model, setting, device):
    
    path = os.path.join(args.checkpoints, setting)

    train_data, train_loader = _get_data(args, flag='train')
    vali_data, vali_loader = _get_data(args, flag='val')
    test_data, test_loader = _get_data(args, flag='test')

    time_now = time.time()

    torch.cuda.empty_cache()

    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model parameters: ", params)

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    if args.scheduler:
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = args.pct_start,
                                            epochs = args.finetune_epochs,
                                            max_lr = args.learning_rate)
    else:
        scheduler = None
   
    best_vali_loss = float('inf')
    best_checkpoint_pretrain_path = os.path.join(path, 'best_model_finetue_pred_len_'+str(args.pred_len)+'.pt')
    for epoch in range(args.finetune_epochs):
        print("Epoch number: ", epoch)
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(train_loader)):
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)

            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            # encoder - decoder
            if 'Linear' in args.model or 'TST' in args.model:
                outputs = model(batch_x)
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 300 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            loss.backward()
            model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        model, vali_loss = vali(args, model, device, vali_data, vali_loader, criterion)
        model, test_loss = vali(args, model, device, test_data, test_loader, criterion)
        
        if vali_loss < best_vali_loss:
            best_vali_loss = vali_loss
            # Save best model
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': model.state_dict(),
                'optimizer_state_dict': model_optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train loss': train_loss,
                'vali_loss': vali_loss,
                'test_loss': test_loss,
            }, best_checkpoint_pretrain_path)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss, test_loss))


        if args.lradj != 'TST':
            adjust_learning_rate(model_optim, scheduler, epoch + 1, args)
        else:
            print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))