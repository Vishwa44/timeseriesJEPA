import argparse
import os
import torch
import random
import numpy as np
# from TimeSeriesJEPA.pretraining import pretrain
from TimeSeriesJEPA.hf_trainer import pretrain

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc
    parser.add_argument('--checkpoints', type=str, default="./checkpoints/", help='path to save model')

    # data loader
    parser.add_argument('--data_path', type=str, default='./Time_300B', help='data file')
    parser.add_argument('--nenc', type=int, default=1, help='number of enc masks')
    parser.add_argument('--npred', type=int, default=3, help='number of pred masks')
    parser.add_argument('--min_keep', type=int, default=5, help='min_keep')
    parser.add_argument('--allow_overlap', type=bool, default=False, help='allow overlap')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=720, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--pre_norm', type=bool, default=True, help='pre norm')
    parser.add_argument('--channel_attention', type=bool, default=False, help='channel attention')
    
    # Formers 
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--num_hidden_layers', type=int, default=5, help='num of encoder layers')
    parser.add_argument('--ffn_dim', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--pretrain_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--train_scale', type=float, default=1.0, help='scale the target encoder')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')

    args = parser.parse_args()

    args.ema = [0.996, 1.0]
    args.enc_mask_scale = [0.85, 1]
    args.pred_mask_scale = [0.15, 0.2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print('Args in experiment:')
    print(args)


    setting = 'PatchTST_Time300B_sl{}_dm{}_nh{}_el{}_fd{}_bs{}_lr{}'.format(
        args.seq_len,
        args.d_model,
        args.n_heads,
        args.num_hidden_layers,
        args.ffn_dim,
        args.batch_size,
        args.learning_rate
        )

    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

    pretrain(args, setting, device)
    torch.cuda.empty_cache()
