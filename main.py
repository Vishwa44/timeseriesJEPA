import argparse
import os
import torch
import random
import numpy as np
from pretraining import pretrain
from finetune import finetune
from load_config import load_config
from model.PatchTST_encoder import PatchTST_embedding


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--pretraining', type=bool, default=1, help='pretrining')
    parser.add_argument('--finetuning', type=bool, default=1, help='pretrining')
    parser.add_argument('--config_path', required=True, type=str, help='config yaml file path')
    # parser.add_argument('--encoder_path', required=True, type=str, help='encoder file path')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction length')
    p_args = parser.parse_args()

    # random seed
    fix_seed = p_args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print('Args in experiment:')
    print(p_args)

    args, setting = load_config(p_args.config_path, p_args.pred_len, p_args.model_id)
    
    enc = None
    if p_args.pretraining:
        enc = pretrain(args, setting, device)
    
    torch.cuda.empty_cache()

    if p_args.finetuning:
        path = os.path.join(args.checkpoints, setting,'best_model_pretrain.pt')
        if path.exists() and enc is None:
            checkpoint = torch.load(path)
            enc = PatchTST_embedding(args).float().to(device)
            enc.load_state_dict(checkpoint['encoder_state_dict'])
        else:
            raise Exception("Load encoder path")
        finetune(args, enc, setting, device)

    