from TimeSeriesJEPA.datasets.time_moe_dataset import TimeMoEDataset
from TimeSeriesJEPA.datasets.time_moe_window_dataset import TimeMoEWindowDataset
import random
from TimeSeriesJEPA.datasets.mask_collator import TimeSeriesMaskCollator
from TimeSeriesJEPA.models.PatchTST import PatchTSTModelJEPA, PatchTSTPredictorModelJEPA
from TimeSeriesJEPA.datasets.mask_utils import apply_masks
from transformers import PatchTSTConfig, Trainer, TrainingArguments
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
    print("dataset loaded, total size: ", len(windowds))
    return windowds


class TimeSeriesJEPATrainer(Trainer):
    def __init__(
        self,
        encoder,
        predictor,
        target_encoder=None,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        **kwargs
    ):
        # Initialize target encoder if not provided
        if target_encoder is None:
            target_encoder = copy.deepcopy(encoder)
            for p in target_encoder.parameters():
                p.requires_grad = False
                
        self.predictor = predictor
        self.target_encoder = target_encoder
        
        # Store original encoder to pass to parent class
        self.original_encoder = encoder
        
        # Initialize momentum scheduler parameters
        self.momentum_start = kwargs.pop('momentum_start', 0.996)
        self.momentum_end = kwargs.pop('momentum_end', 1.0)
        self.current_momentum = self.momentum_start
        
        # Initialize parent Trainer
        super().__init__(
            model=encoder,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs
        )

    def create_optimizer(self):
        """
        Custom optimizer creation with parameter groups
        """
        param_groups = [
            {
                'params': (p for n, p in self.model.named_parameters()
                          if ('bias' not in n) and (len(p.shape) != 1))
            },
            {
                'params': (p for n, p in self.predictor.named_parameters()
                          if ('bias' not in n) and (len(p.shape) != 1))
            },
            {
                'params': (p for n, p in self.model.named_parameters()
                          if ('bias' in n) or (len(p.shape) == 1)),
                'weight_decay': 0
            },
            {
                'params': (p for n, p in self.predictor.named_parameters()
                          if ('bias' in n) or (len(p.shape) == 1)),
                'weight_decay': 0
            }
        ]
        
        self.optimizer = torch.optim.Adam(
            param_groups,
            lr=self.args.learning_rate
        )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation implementing JEPA training logic
        """
        seq_x, enc_masks, pred_masks = inputs[0], inputs[-2], inputs[-1]
        
        # Forward pass through target encoder
        with torch.no_grad():
            h = self.target_encoder(seq_x)
            h = F.layer_norm(h[0], (h[0].size(-1),))
            h = self._apply_masks(h, pred_masks)
        
        # Forward pass through encoder and predictor
        z = model(seq_x, enc_masks)
        z = self.predictor(z[0], enc_masks, pred_masks)
        
        # Compute loss
        loss = F.smooth_l1_loss(z[0], h)
        
        # Update target encoder with momentum
        self._momentum_update()
            
        return (loss, None) if return_outputs else loss

    def _momentum_update(self):
        """
        Update target encoder parameters using momentum
        """
        # Calculate current momentum based on training progress
        progress = self.state.global_step / (len(self.train_dataset) * self.args.num_train_epochs)
        self.current_momentum = self.momentum_start + (self.momentum_end - self.momentum_start) * progress
        
        # Update target encoder parameters
        with torch.no_grad():
            for param_q, param_k in zip(self.model.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(self.current_momentum).add_((1 - self.current_momentum) * param_q.detach().data)

    def _apply_masks(self, h, masks):
        """
        Apply prediction masks to hidden states
        """
        return apply_masks(h, masks)
    
    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "shuffle": False
        }


        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


def pretrain(args, setting, device):
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
        
    train_data = _get_data(args, collator=mask_collator)

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

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.pretrain_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_strategy="epoch",
        max_steps=1000,
        logging_strategy="steps",
        logging_steps=10
    )

    # Initialize the trainer
    trainer = TimeSeriesJEPATrainer(
        encoder=encoder,
        predictor=predictor,
        target_encoder=target_encoder,
        args=training_args,
        train_dataset=train_data,
        data_collator=mask_collator,
        momentum_start=args.ema[0],
        momentum_end=args.ema[1],
    )

    # Train the model
    trainer.train()