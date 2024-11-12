from data_provider.data_factory import data_provider
from data_provider.mask_collator import TimeSeriesMaskCollator
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from model.PatchTST_encoder import PatachTST_embedding
from model.PatchTST_predictor import PatachTST_predictor
from model.PatchTST_finetune import PatchTST_finetune
from data_provider.mask_utils import apply_masks

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import time
import copy
import os
import time

from tqdm import tqdm
def _get_data(args, flag, collator=None):
        data_set, data_loader = data_provider(args, flag, collator)
        return data_set, data_loader

def pretrain(args, device):
    
    mask_collator = TimeSeriesMaskCollator(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            patch_size=args.patch_len,
            stride=args.stride,
            pred_mask_scale=args.pred_mask_scale,
            enc_mask_scale=args.enc_mask_scale,
            nenc=args.nenc,
            npred=args.npred,
            allow_overlap=args.allow_overlap,
            min_keep=args.min_keep)

    train_data, train_loader = _get_data(args, flag='train', collator=mask_collator)

    encoder = PatachTST_embedding(args).float().to(device)
    predictor = PatachTST_predictor(args).float().to(device)

    target_encoder = copy.deepcopy(encoder)
    model_parameters = filter(lambda p: p.requires_grad, encoder.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("encoder parameters: ", params)
    model_parameters = filter(lambda p: p.requires_grad, predictor.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("predictor parameters: ", params)

    train_steps = len(train_loader)

    param_groups = [
            {
                'params': (p for n, p in encoder.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in predictor.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in encoder.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }, {
                'params': (p for n, p in predictor.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }

        ]
    model_optim = optim.Adam(param_groups, lr=args.learning_rate)
        
    for p in target_encoder.parameters():
        p.requires_grad = False

    momentum_scheduler = (args.ema[0] + i*(args.ema[1]-args.ema[0])/(train_steps*args.train_epochs*args.train_scale)
                            for i in range(int(train_steps*args.train_epochs*args.train_scale)+1))

    scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                                steps_per_epoch = train_steps,
                                                pct_start = args.pct_start,
                                                epochs = args.train_epochs,
                                                max_lr = args.learning_rate)

    for epoch in range(args.train_epochs):
        print("Epoch number: ", epoch)
        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        for i, (seq_x, seq_y, seq_x_mark, seq_y_mark, enc_masks, pred_masks) in enumerate(tqdm(train_loader)):
            iter_count += 1
            seq_x = seq_x.float().to(device)
            enc_masks = [u.to(device, non_blocking=True) for u in enc_masks]
            pred_masks = [u.to(device, non_blocking=True) for u in pred_masks]
            def train_step():
                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(seq_x)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, pred_masks)
                        return h

                def forward_context():
                    z = encoder(seq_x, enc_masks)
                    z = predictor(z, enc_masks, pred_masks)
                    return z

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    return loss

                # Step 1. Forward
                h = forward_target()
                z = forward_context()
                loss = loss_fn(z, h)

                #  Step 2. Backward & step

                loss.backward()
                model_optim.step()
                model_optim.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return float(loss)
            loss = train_step()
        train_loss.append(loss)
        adjust_learning_rate(model_optim, scheduler, epoch + 1, args)       
        train_loss = np.average(train_loss)
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
            epoch + 1, train_steps, train_loss))
        
    return encoder