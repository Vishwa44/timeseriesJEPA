from data_provider.data_factory import data_provider
from data_provider.mask_collator import TimeSeriesMaskCollator
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from model.PatchTST_encoder import PtachTST_embedding
from model.PatchTST_predictor import PtachTST_predictor
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

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class config_etth1_patchtst():
    def __init__(self, name="", seq_len=512, pred_len=96, num_epochs=1) -> None:
        self.model_type = "PatchTST"
        self.is_training = 1
        self.model_id = "PatchTST_attn_Etth1_"+str(seq_len)+"_"+str(pred_len)+"_"+name
        self.model = "PatchTST"
        self.data = "ETTh1"
        self.root_path = r"D:\Coursework\MTS\dataset\ETT-small"
        self.data_path = "ETTh1.csv"
        self.features = "M"
        self.target = "OT"
        self.freq = "h"
        self.checkpoints = "./checkpoints/"
        self.seq_len = seq_len
        self.label_len = 48
        self.pred_len = pred_len
        self.fc_dropout = 0.2
        self.head_dropout = 0.0
        self.patch_len = 16
        self.stride = 8
        self.padding_patch = "end"
        self.affine = 0
        self.subtract_last = 0
        self.decomposition = 0
        self.kernel_size = 25
        self.individual = 0
        self.embed_type = 0
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 16
        self.predictor_d_model = 16
        self.revin = 1
        self.n_heads = 4
        self.e_layers = 3
        self.d_layers = 1
        self.d_ff = 128
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.3 # 0.2
        self.fusion_dropout = 0.3
        self.proj_dropout = 0.3
        self.embed = "timeF"
        self.activation = "gelu"
        self.output_attention = False
        self.do_predict = False
        self.num_workers = 2
        self.itr = 1
        self.train_epochs = num_epochs
        self.batch_size = 128
        self.patience = 50
        self.learning_rate = 0.0001
        self.des = "Exp"
        self.loss = "mse"
        self.lradj = "type3"
        self.pct_start = 0.3
        self.use_amp = False
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'
        self.test_flop = False
        self.profile = False
        self.scheduler = True
        self.use_norm = True
        self.embedding_model = True
        # jepa
        self.enc_mask_scale=(0.85, 1)
        self.pred_mask_scale=(0.15, 0.2)
        self.use_embed = True
        self.nenc=1
        self.npred=3
        self.allow_overlap=False
        self.min_keep=5
        self.embedding_model = True
        self.ema = [0.996, 1.0]
        self.train_scale = 1.0
        pass

class config_elec_patchtst():
    def __init__(self, name="", seq_len=512, pred_len=96, num_epochs=1) -> None:
        self.model_type = "PatchTST"
        self.is_training = 1
        self.model_id = "PatchTST_attn_Electricity_"+str(seq_len)+"_"+str(pred_len)+"_"+name
        self.model = "PatchTST"
        self.data = "custom"
        self.root_path = r"D:\Coursework\MTS\dataset"
        self.data_path = "electricity.csv"
        self.features = "M"
        self.target = "OT"
        self.freq = "h"
        self.checkpoints = "./checkpoints/"
        self.seq_len = seq_len
        self.label_len = 48
        self.pred_len = pred_len
        self.fc_dropout = 0.2
        self.head_dropout = 0.0
        self.patch_len = 16
        self.stride = 8
        self.padding_patch = "end"
        self.revin = 1
        self.affine = 0
        self.subtract_last = 0
        self.decomposition = 0
        self.kernel_size = 25
        self.individual = 0
        self.embed_type = 0
        self.enc_in = 321
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 64 #128
        self.n_heads = 16
        self.e_layers = 3
        self.d_layers = 1
        self.d_ff = 128 # 256
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.2 # 0.2
        self.fusion_dropout = 0.05
        self.proj_dropout = 0.05
        self.embed = "timeF"
        self.activation = "gelu"
        self.output_attention = False
        self.do_predict = False
        self.num_workers = 8
        self.itr = 1
        self.train_epochs = num_epochs
        self.batch_size = 16
        self.patience = 50
        self.learning_rate = 0.0001
        self.des = "Exp"
        self.loss = "mse"
        self.lradj = "TST"
        self.pct_start = 0.2
        self.use_amp = False
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'
        self.test_flop = False
        self.profile = False
        self.scheduler = True
        random_seed=2021
        # JEPA
        self.predictor_d_model = 64 # 128
        self.enc_mask_scale=(0.85, 1)
        self.pred_mask_scale=(0.15, 0.2)
        self.use_embed = True
        self.nenc=1
        self.npred=3
        self.allow_overlap=False
        self.min_keep=5
        self.embedding_model = True
        self.ema = [0.996, 1.0]
        self.train_scale = 1.0
        pass

class config_ettm1_patchtst():
    def __init__(self ,name="", seq_len=512, pred_len=96, num_epochs=1) -> None:
        self.model_type = "PatchTST"
        self.is_training = 1
        self.model_id = "PatchTST_attn_ETTm1_"+str(seq_len)+"_"+str(pred_len)+"_"+name
        self.model = "PatchTST"
        self.data = "ETTm1"
        self.root_path = r"D:\Coursework\MTS\dataset\ETT-small"
        self.data_path = "ETTm1.csv"
        self.features = "M"
        self.target = "OT"
        self.freq = "h"
        self.checkpoints = "./checkpoints/"
        self.seq_len = seq_len
        self.label_len = 48
        self.pred_len = pred_len
        self.fc_dropout = 0.2
        self.head_dropout = 0.0
        self.patch_len = 16
        self.stride = 8
        self.padding_patch = "end"
        self.revin = 1
        self.affine = 0
        self.subtract_last = 0
        self.decomposition = 0
        self.kernel_size = 25
        self.individual = 0
        self.embed_type = 0
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 128
        self.n_heads = 16
        self.e_layers = 3
        self.d_layers = 1
        self.d_ff = 256
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.2 # 0.2
        self.fusion_dropout = 0.2
        self.proj_dropout = 0.2
        self.embed = "timeF"
        self.activation = "gelu"
        self.output_attention = False
        self.do_predict = False
        self.num_workers = 8
        self.itr = 1
        self.train_epochs = num_epochs
        self.batch_size = 128
        self.patience = 50
        self.learning_rate = 0.0001
        self.des = "Exp"
        self.loss = "mse"
        self.lradj = "TST"
        self.pct_start = 0.4
        self.use_amp = False
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'
        self.test_flop = False
        self.profile = False
        self.scheduler = True
        self.use_norm = True
        # JEPA
        self.predictor_d_model = 128 # 128
        self.enc_mask_scale=(0.85, 1)
        self.pred_mask_scale=(0.15, 0.2)
        self.use_embed = True
        self.nenc=1
        self.npred=3
        self.allow_overlap=False
        self.min_keep=5
        self.embedding_model = True
        self.ema = [0.996, 1.0]
        self.train_scale = 1.0
        pass


def _get_data(args, flag, collator=None):
        data_set, data_loader = data_provider(args, flag, collator)
        return data_set, data_loader

def pretrain(args):
    
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

    encoder = PtachTST_embedding(args).float().to(device)
    predictor = PtachTST_predictor(args).float().to(device)

    target_encoder = copy.deepcopy(encoder)
    model_parameters = filter(lambda p: p.requires_grad, encoder.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("encoder parameters: ", params)
    model_parameters = filter(lambda p: p.requires_grad, predictor.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("predictor parameters: ", params)

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

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

def train(args, model):
    train_data, train_loader = _get_data(args, flag='train')
    vali_data, vali_loader = _get_data(args, flag='val')
    test_data, test_loader = _get_data(args, flag='test')



    time_now = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
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
                                            epochs = args.train_epochs,
                                            max_lr = args.learning_rate)
    else:
        scheduler = None

    for epoch in range(args.train_epochs):
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

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss, test_loss))


        if args.lradj != 'TST':
            adjust_learning_rate(model_optim, scheduler, epoch + 1, args)
        else:
            print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))


if __name__ == '__main__':
    fix_seed = 2023
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args = config_ettm1_patchtst(num_epochs=50)
    encoder = pretrain(args)

    for param in encoder.parameters():
        param.requires_grad = False

    model = PatchTST_finetune(args, encoder).float().to(device)
    train(args, model)