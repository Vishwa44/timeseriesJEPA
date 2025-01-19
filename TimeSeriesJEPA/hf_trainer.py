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
import wandb

os.environ["WANDB_PROJECT"] = "TimeSeriesJEPA" 


def _get_data(args, collator):
    trainds = TimeMoEDataset(args.data_path, val=False)
    trainwindowds = TimeMoEWindowDataset(trainds, context_length=args.seq_len, prediction_length=0)
    valds = TimeMoEDataset(args.data_path, val=True)
    valwindowds = TimeMoEWindowDataset(valds, context_length=args.seq_len, prediction_length=0)
    print("dataset loaded, total size: ", len(trainwindowds))
    return trainwindowds, valwindowds


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
        self.predictor.train()

        self.target_encoder = target_encoder
        
        # Store original encoder to pass to parent class
        self.original_encoder = encoder
        
        # Initialize momentum scheduler parameters
        self.momentum_start = kwargs.pop('momentum_start', 0.996)
        self.momentum_end = kwargs.pop('momentum_end', 1.0)
        self.train_scale = kwargs.pop('train_scale', 1.0)
        self.current_momentum = self.momentum_start
        
        # Initialize parent Trainer
        super().__init__(
            model=encoder,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs
        )
        self.momentum_scheduler = (self.momentum_start + i*(self.momentum_end-self.momentum_start)/(len(self.train_dataset)*self.args.num_train_epochs*self.train_scale)
                            for i in range(int(len(self.train_dataset) * self.args.num_train_epochs*self.train_scale)+1))

    def create_optimizer(self):
        """
        Custom optimizer creation with parameter groups
        """
        print("custom optimizer created")
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
    
    def create_scheduler(self, num_training_steps, optimizer):
        if self.lr_scheduler is None:
            self.lr_scheduler = lr_scheduler.OneCycleLR(optimizer = optimizer,
                                                    steps_per_epoch = num_training_steps,
                                                    pct_start = 0.1,
                                                    max_lr = self.args.learning_rate, epochs = self.args.num_train_epochs)
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation implementing JEPA training logic
        """
        seq_x, enc_masks, pred_masks = inputs[0], inputs[-2], inputs[-1]
        
        def train_step():
            def forward_target():
                with torch.no_grad():
                    h = self.target_encoder(seq_x)
                    h = F.layer_norm(h[0], (h[0].size(-1),))  # normalize over feature-dim
                    B = len(h[0])
                    # -- create targets (masked regions of h)
                    h = self._apply_masks(h, pred_masks)
                    return h

            def forward_context():
                z = model(seq_x, enc_masks)
                z = self.predictor(z[0], enc_masks, pred_masks)
                return z[0]

            def loss_fn(z, h):
                loss = F.smooth_l1_loss(z, h)
                return loss

            # Step 1. Forward
            h = forward_target()
            z = forward_context()
            loss = loss_fn(z, h)

            return loss
        loss = train_step()    
        return (loss, None) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()
        
        kwargs = {}

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        
        self.accelerator.backward(loss, **kwargs)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._momentum_update()
        # Finally we need to normalize the loss for reporting
        if num_items_in_batch is None:
            return loss.detach() / self.args.gradient_accumulation_steps
        return loss.detach()
    
    def _momentum_update(self):
        """
        Update target encoder parameters using momentum
        """
        m = next(self.momentum_scheduler)
        with torch.no_grad():
            for param_q, param_k in zip(self.model.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

    def _apply_masks(self, h, masks):
        """
        Apply prediction masks to hidden states
        """
        return apply_masks(h, masks)
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys = None):
        """
        Custom prediction step for evaluation that reuses compute_loss.
        Returns tuple of (loss, predictions, labels)
        """
        with torch.no_grad():  # Disable gradients for evaluation
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            
            if prediction_loss_only:
                return (loss, None, None)
            
            # Get the predictions and targets from compute_loss outputs if needed
            predictions = outputs[0] if outputs is not None else None
            labels = outputs[1] if outputs is not None and len(outputs) > 1 else None
            
            return (loss, predictions, labels)
        
    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "shuffle": True
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
        
    train_data, val_data = _get_data(args, collator=mask_collator)

    enc_config = PatchTSTConfig(
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
    pred_config = PatchTSTConfig(
                        num_input_channels=1,
                        context_length=args.seq_len,
                        patch_length=args.patch_len,
                        patch_stride=args.stride,
                        prediction_length=96,
                        random_mask_ratio=0.4,
                        d_model=args.pred_d_model,
                        num_attention_heads=args.pred_n_heads,
                        num_hidden_layers=args.pred_num_hidden_layers,
                        ffn_dim=args.pred_ffn_dim,
                        dropout=args.pred_dropout,
                        head_dropout=args.pred_head_dropout,
                        pooling_type=None,
                        channel_attention=False,
                        scaling="std",
                        pre_norm=args.pre_norm,
                        norm_type="batchnorm",
                        positional_encoding_type = "sincos"
                        )
    
    encoder = PatchTSTModelJEPA(enc_config).float().to(device)
    predictor = PatchTSTPredictorModelJEPA(pred_config).float().to(device)

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
        max_steps=1000000,
        logging_strategy="steps",
        logging_steps=100,
        do_eval = True,
        eval_strategy="steps",                                     
        eval_steps=100000,
        report_to="wandb"         
    )                                                                                                                                                                                                                                        
                                                                                                                                                                                                                     
    # Initialize the trainer                                                            
    trainer = TimeSeriesJEPATrainer(
        encoder=encoder,
        predictor=predictor,
        target_encoder=target_encoder,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=mask_collator,
        momentum_start=args.ema[0],
        momentum_end=args.ema[1],
    )

    # Train the model
    trainer.train()
    trainer.predictor.save_pretrained(os.path.join(args.checkpoint, "predictor"), safe_serialization=False)