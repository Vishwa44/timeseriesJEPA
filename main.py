import argparse
import os
import torch
import random
import numpy as np
from pretraining import pretrain
from finetune import finetune
from load_model_config import load_model_config
from model.PatchTST_finetune import PatchTST_finetune
from model.PatchTST_encoder import PatchTST_embedding


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--pretraining', action='store_true', help='pretrining')
    parser.add_argument('--finetuning', action='store_true', help='finetuning')
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

    args, setting = load_model_config(p_args.config_path, p_args.pred_len, p_args.model_id)
    
    if p_args.pretraining:
        pretrain(args, setting, device)
    
    torch.cuda.empty_cache()
    print("setting: ", setting)
    if p_args.finetuning:
        path = os.path.join(args.checkpoints, setting,'best_model_pretrain.pt')
        if os.path.exists(path):
            checkpoint = torch.load(path)
            enc = PatchTST_embedding(args).float().to(device)
            enc.load_state_dict(checkpoint['encoder_state_dict'])
            for param in enc.parameters():
                param.requires_grad = False

            model = PatchTST_finetune(args, enc).float().to(device)
        else:
            raise Exception("Load encoder path")
        finetune(args, model, setting, device)

    