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
    parser.add_argument('--pretraining_checkpoints', type=str, default="./results/", help='path to save model')

    # data loader
    parser.add_argument('--dataset', type=str, default="Time300B", help='training dataset')
    parser.add_argument('--data_path', type=str, default='./Time_300B', help='data file')
    parser.add_argument('--nenc', type=int, default=1, help='number of enc masks')
    parser.add_argument('--npred', type=int, default=4, help='number of pred masks')
    parser.add_argument('--min_keep', type=int, default=5, help='min_keep')
    parser.add_argument('--allow_overlap', type=bool, default=False, help='allow overlap')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=720, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # PatchTST encoder
    parser.add_argument('--patch_len', type=int, default=8, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--pre_norm', type=bool, default=True, help='pre norm')
    parser.add_argument('--channel_attention', type=bool, default=False, help='channel attention')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--num_hidden_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--ffn_dim', type=int, default=64, help='dimension of fcn')
    parser.add_argument('--norm_type', type=str, default='batchnorm',help='Normalization type')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--compress_proj', type=bool, default=False, help='adds a compressive projection layer at the end of the model')
    parser.add_argument('--compress_proj_size', type=int, default=32, help='size of compressive projection layer')

    # Predictor
    parser.add_argument('--enc_dim', type=int, default=64, help='projection dimension of model')
    parser.add_argument('--pred_pre_norm', type=bool, default=True, help='pre norm')
    parser.add_argument('--pred_d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--pred_n_heads', type=int, default=2, help='num of heads')
    parser.add_argument('--pred_num_hidden_layers', type=int, default=1, help='num of predictor layers')
    parser.add_argument('--pred_ffn_dim', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--pred_dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--pred_norm_type', type=str, default='batchnorm',help='Normalization type')

    # optimization
    parser.add_argument('--reg_coeff', type=int, default=0.0000, help='regularization coff')
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--pretrain_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--max_steps', type=int, default=3000, help='max training steps')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--train_scale', type=float, default=1.0, help='scale the target encoder')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')

    #logging
    parser.add_argument('--logging_steps', type=int, default=100, help='Print train loss after how many steps')
    parser.add_argument('--eval_steps', type=int, default=1000, help='Eval steps')

    args = parser.parse_args()

    args.ema = [0.96, 1.0]
    # args.enc_mask_scale = [0.85, 1]
    args.enc_mask_scale = [0.45, 0.6]
    args.pred_mask_scale = [0.15, 0.2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print('Args in experiment:')
    print(args)
    
    setting = 'PatchTST_{}_sl{}_enc_dm{}_nh{}_el{}_fd{}_pred_dm{}_nh{}_el{}_fd{}_bs{}_lr{}_pe{}_data_nenc{}_npred{}_new_collator'.format(
        args.dataset,
        args.seq_len,
        args.d_model,
        args.n_heads,
        args.num_hidden_layers,
        args.ffn_dim,
        args.pred_d_model,
        args.pred_n_heads,
        args.pred_num_hidden_layers,
        args.pred_ffn_dim,
        args.batch_size,
        args.learning_rate,
        args.pretrain_epochs,
        args.nenc,
        args.npred
        )
    
    if args.compress_proj:
        setting += "_compress_"+str(args.compress_proj_size)
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

    pretrain(args, setting, device)
    torch.cuda.empty_cache()
