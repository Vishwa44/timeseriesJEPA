from data_provider.data_factory import data_provider
from data_provider.mask_collator import TimeSeriesMaskCollator
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')
class config_etth1_patchtst():
    def __init__(self, name, seq_len, pred_len, num_epochs) -> None:
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
        self.num_workers = 1
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
        self.red_mask_scale=(0.15, 0.2)
        self.use_embed = True
        self.nenc=1
        self.npred=2
        self.allow_overlap=False
        self.min_keep=5
        self.embedding_model = True
        pass


def _get_data(args, flag, collator=None):
        data_set, data_loader = data_provider(args, flag, collator)
        return data_set, data_loader

def train(args):
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
    vali_data, vali_loader = _get_data(args, flag='val', collator=mask_collator)
    test_data, test_loader = _get_data(args, flag='test', collator=mask_collator)
    path = os.path.join(args.checkpoints, args.model_id)
    if not os.path.exists(path):
        os.makedirs(path)

    writer = SummaryWriter(log_dir=path)

    time_now = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.cuda.empty_cache()

    # model = patchtst_model(args).float().to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model parameters: ", params)

    train_steps = len(train_loader)
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


if __name__ == '__main__':
    fix_seed = 2023
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    num_epochs = 50
    args = config_etth1_patchtst(num_epochs=num_epochs)
    train(args)

    