#!/usr/bin/env python
# coding: utf-8

# # import libraries, load data

import numpy as np
import torch 
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import pdb

class lstm(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, in_seq_len, out_seq_len, in_factors):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True) 
        self.fc = nn.Linear(in_seq_len*hidden_dim, output_dim*out_seq_len) 
        self.tanH = nn.Tanh()
        self.num_out_feats = output_dim  
        self.out_seq_len = out_seq_len
        self.in_factors = in_factors

    def forward(self, x):
        B, T, _ = x.shape
        x = self.normalize(x)
        out, hidden = self.lstm(x)
        out = out.flatten(start_dim=1)
        out = self.tanH(out)
        out = self.fc(out)
        out = torch.reshape(out, [B, self.out_seq_len, self.num_out_feats])
        return out
    
    def normalize(self, x):
        ## x should be of shape B, T, N_feats == in_factors.shape[1]
        self.in_factors = (self.in_factors.to("cuda") if x.is_cuda else self.in_factors.to("cpu"))
        means = self.in_factors[0]
        std = self.in_factors[1]

        return (x-means)/std



