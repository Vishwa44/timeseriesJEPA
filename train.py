from data_provider.data_factory import data_provider
from data_provider.mask_collator import TimeSeriesMaskCollator
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from model.PatchTST_encoder import PatachTST_embedding
from model.PatchTST_predictor import PatachTST_predictor
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

from pretraining import pretrain
from finetune import finetune

warnings.filterwarnings('ignore')



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

class config_etth2_patchtst():
    def __init__(self ,name="", seq_len=512, pred_len=96, num_epochs=1) -> None:
        self.model_type = "PatchTST"
        self.is_training = 1
        self.model_id = "PatchTST_attn_Etth2_"+str(seq_len)+"_"+str(pred_len)+"_"+name
        self.model = "PatchTST"
        self.data = "ETTh2"
        self.root_path = r"D:\Coursework\MTS\dataset\ETT-small"
        self.data_path = "ETTh2.csv"
        self.features = "M"
        self.target = "OT"
        self.freq = "h"
        self.checkpoints = "./checkpoints/"
        self.seq_len = seq_len
        self.label_len = 48
        self.pred_len = pred_len
        self.fc_dropout = 0.3
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
        self.d_model = 16
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
        self.num_workers = 8
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
        # JEPA
        self.predictor_d_model = 16 # 128
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

class config_weather_patchtst():
    def __init__(self, name="", seq_len=512, pred_len=96, num_epochs=1) -> None:
        self.model_type = "PatchTST"
        self.is_training = 1
        self.model_id = "PatchTST_attn_Etth1_"+str(seq_len)+"_"+str(pred_len)+"_"+name
        self.model = "PatchTST"
        self.data = "custom"
        self.root_path = "/home/vg2523/datasets"
        self.data_path = "weather.csv"
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
        self.enc_in = 21
        self.dec_in = 21
        self.c_out = 7
        self.d_model = 128
        self.predictor_d_model = 128
        self.revin = 1
        self.n_heads = 16
        self.e_layers = 3
        self.d_layers = 1
        self.d_ff = 256
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.2 # 0.2
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



if __name__ == '__main__':
    fix_seed = 2023
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args = config_etth2_patchtst(num_epochs=50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = pretrain(args, device)

    for param in encoder.parameters():
        param.requires_grad = False

    model = PatchTST_finetune(args, encoder).float().to(device)
    finetune(args, model, device)