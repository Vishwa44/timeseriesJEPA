from datasets.time_moe_dataset import TimeMoEDataset
from datasets.time_moe_window_dataset import TimeMoEWindowDataset
import random
from datasets.mask_collator import TimeSeriesMaskCollator
from models.PatchTST import PatchTSTModelJEPA, PatchTSTPredictorModelJEPA
from datasets.mask_utils import apply_masks
from transformers import PatchTSTConfig
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
import copy
import os
import time
from pathlib import Path
from accelerate import Accelerator
from tqdm import tqdm

def _get_data(args, collator):
    ds = TimeMoEDataset(args.data_path)
    windowds = TimeMoEWindowDataset(ds, context_length=args.seq_len, prediction_length=0)
    data_loader = DataLoader(
        windowds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collator
        )
    print("dataset loaded, total size: ", len(windowds))
    return windowds, data_loader

def pretrain(args, setting, device):
    
    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
            os.makedirs(path)

    mask_collator = TimeSeriesMaskCollator(
            seq_len=args.seq_len,
            patch_size=args.patch_len,
            stride=args.stride,
            pred_mask_scale=args.pred_mask_scale,
            enc_mask_scale=args.enc_mask_scale,
            nenc=args.nenc,
            npred=args.npred,
            allow_overlap=args.allow_overlap,
            min_keep=args.min_keep)

    train_data, train_loader = _get_data(args, collator=mask_collator)

    config = PatchTSTConfig(
                        num_input_channels=1,
                        context_length=args.seq_len,
                        patch_length=args.patch_len,
                        patch_stride=args.stride,
                        prediction_length=96,
                        random_mask_ratio=0.4,
                        d_model=args.d_model,
                        num_attention_heads=args.n_heads,
                        num_hidden_layers=args.num_hidden_layers,
                        ffn_dim=args.ffn_dim,
                        dropout=args.dropout,
                        head_dropout=args.head_dropout,
                        pooling_type=None,
                        channel_attention=args.channel_attention,
                        scaling="std",
                        pre_norm=args.pre_norm,
                        norm_type="batchnorm",
                        positional_encoding_type = "sincos"
                        )
    
    encoder = PatchTSTModelJEPA(config).float().to(device)
    predictor = PatchTSTPredictorModelJEPA(config).float().to(device)

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

    momentum_scheduler = (args.ema[0] + i*(args.ema[1]-args.ema[0])/(train_steps*args.pretrain_epochs*args.train_scale)
                            for i in range(int(train_steps*args.pretrain_epochs*args.train_scale)+1))

    # scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
    #                                             steps_per_epoch = train_steps,
    #                                             pct_start = args.pct_start,
    #                                             epochs = args.pretrain_epochs,
    #                                             max_lr = args.learning_rate)

    start_epoch = 0
    best_loss = float('inf')

    for epoch in range(start_epoch, args.pretrain_epochs):
        print("Epoch number: ", epoch)
        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        for i, batch in enumerate(tqdm(train_loader)):
            seq_x, enc_masks, pred_masks = batch[0], batch[-2], batch[-1]
            iter_count += 1
            seq_x = seq_x.float().to(device)
            enc_masks = [u.to(device, non_blocking=True) for u in enc_masks]
            pred_masks = [u.to(device, non_blocking=True) for u in pred_masks]
            def train_step():
                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(seq_x)
                        h = F.layer_norm(h[0], (h[0].size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, pred_masks)
                        return h

                def forward_context():
                    z = encoder(seq_x, enc_masks)
                    z = predictor(z[0], enc_masks, pred_masks)
                    return z[0]

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
            if i%10 == 0:
                print("Train loss so far: ", np.average(train_loss))
        epoch_loss = np.average(train_loss)
        

        # adjust_learning_rate(model_optim, scheduler, epoch + 1, args)       
        train_loss = np.average(train_loss)
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
            epoch + 1, train_steps, train_loss))
        

