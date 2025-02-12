import argparse
import os
import torch
import random
import numpy as np
# from TimeSeriesJEPA.pretraining import pretrain
from TimeSeriesJEPA.hf_trainer import finetune

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--finetuning_checkpoints', type=str, default="./finetuning_results/", help='path to save model')
    parser.add_argument('--dataset', type=str, default="Time300B", help='training dataset')
    parser.add_argument('--data_path', type=str, default='./Time_300B', help='data file')
    parser.add_argument('--model_path', type=str, default='.', help='model path')
    
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=720, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # PatchTST encoder
    parser.add_argument('--patch_len', type=int, default=8, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--finetune_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--max_steps', type=int, default=3000, help='max training steps')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=1e-9, help='optimizer learning rate')

    #logging
    parser.add_argument('--logging_steps', type=int, default=100, help='Print train loss after how many steps')
    parser.add_argument('--eval_steps', type=int, default=1000, help='Eval steps')

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print('Args in experiment:')
    print(args)

    setting = args.model_path.split('\\')[-2]
    setting+="_pl"+str(args.pred_len)
    print('>>>>>>>start finetuning : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting)) 

    finetune(args, setting, device)
    torch.cuda.empty_cache()