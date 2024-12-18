import yaml
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os

@dataclass
class Config:
    model_type: str
    model: str
    model_id: str
    data: str
    task: str
    root_path: str
    data_path: str
    features: str
    target: str
    freq: str
    checkpoints: str
    seq_len: int
    label_len: int
    pred_len: int
    fc_dropout: float
    head_dropout: float
    patch_len: int
    stride: int
    padding_patch: str
    revin: int
    affine: int
    subtract_last: int
    decomposition: int
    kernel_size: int
    individual: int
    embed_type: int
    enc_in: int
    dec_in: int
    c_out: int
    d_model: int
    n_heads: int
    e_layers: int
    d_layers: int
    d_ff: int
    moving_avg: int
    factor: int
    distil: bool
    dropout: float
    embed: str
    activation: str
    output_attention: bool
    do_predict: bool
    num_workers: int
    itr: int
    pretrain_epochs: int
    finetune_epochs: int
    batch_size: int
    patience: int
    learning_rate: float
    des: str
    loss: str
    lradj: str
    pct_start: float
    use_amp: bool
    use_gpu: bool
    gpu: int
    use_multi_gpu: bool
    devices: str
    test_flop: bool
    profile: bool
    scheduler: bool
    use_norm: bool
    predictor_d_model: int
    enc_mask_scale: Tuple[float, float]
    pred_mask_scale: Tuple[float, float]
    use_embed: bool
    nenc: int
    npred: int
    allow_overlap: bool
    min_keep: int
    embedding_model: bool
    ema: List[float]
    train_scale: float

def load_model_config(config_path: str, pred_len: Optional[int] = None, model_id: str = "") -> Config:

    
    # Load YAML file
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    if pred_len is not None:
            config_dict['pred_len'] = pred_len

    
    config_dict['enc_mask_scale'] = tuple(config_dict['enc_mask_scale'])
    config_dict['pred_mask_scale'] = tuple(config_dict['pred_mask_scale'])
    config_dict['model_id'] = model_id

    args = Config(**config_dict)
    setting = '{}_{}_{}_{}_ft{}_sl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                args.model_id,
                args.task,
                args.model,
                args.data_path,
                args.features,
                args.seq_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des)
    print("----"+setting+"----")
    
    return args, setting