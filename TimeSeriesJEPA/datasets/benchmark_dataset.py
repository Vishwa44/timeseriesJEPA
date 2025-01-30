#!/usr/bin/env python
# -*- coding:utf-8 _*-
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class BenchmarkDataset(Dataset):
    def __init__(self, csv_path, context_length: int, prediction_length: int, flag: str, returndict: bool):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.returndict = returndict

        df = pd.read_csv(csv_path)
        
        # Get all columns except date
        cols = df.columns[1:]
        
        base_name = os.path.basename(csv_path).lower()
        if 'etth' in base_name:
            border1s = [0, 12 * 30 * 24 - context_length, 12 * 30 * 24 + 4 * 30 * 24 - context_length]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif 'ettm' in base_name:
            border1s = [0, 12 * 30 * 24 * 4 - context_length, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - context_length]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            num_train = int(len(df) * 0.7)
            num_test = int(len(df) * 0.2)
            num_vali = len(df) - num_train - num_test
            border1s = [0, num_train - context_length, len(df) - num_test - context_length]
            border2s = [num_train, num_train + num_vali, len(df)]


        df_values = df[cols].values
        print("Total data size: ", df_values.shape)
        train_data = df_values[border1s[0]:border2s[0]]

        # Scale the entire dataset at once
        scaler = StandardScaler()
        scaler.fit(train_data)
        scaled_data = scaler.transform(df_values)
        
        if flag == 'test':
            scaled_data = scaled_data[border1s[2]:border2s[2]]
        else:
            scaled_data = scaled_data[border1s[0]:border2s[0]]
            
        # Store the scaled data
        self.data = scaled_data
        self.window_length = self.context_length + self.prediction_length

        # Create indices for all possible windows
        self.indices = []
        n_samples = len(self.data)
        for i in range(n_samples - self.window_length + 1):
            self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.window_length
        
        # Extract the window for all features
        window = self.data[start_idx:end_idx]
        
        # Split into context and prediction windows
        context_window = window[:self.context_length]
        prediction_window = window[self.context_length:]
        if self.returndict:
            return {
            'past_values': context_window.astype(np.float32),  # Shape: [context_length, num_features]
            'future_values': prediction_window.astype(np.float32),  # Shape: [prediction_length, num_features]
        }
        else:
            return context_window.astype(np.float32)
        