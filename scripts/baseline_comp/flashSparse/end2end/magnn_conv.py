#!/usr/bin/env python3
import torch
import sys
import math
import time 
import torch.nn as nn
import torch.nn.functional as F
from tqdm.std import tqdm
import numpy as np
import FS_SDDMM
import FS_SpMM

class MAGNNSpmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, att, X_prime, inputInfo):
        ctx.inputInfo = inputInfo    
        ctx.X_prime = X_prime
        ctx.att = att
        
        X_prime  = FS_SpMM.forward_fp16_gnn(   
        inputInfo.row_pointers, 
        inputInfo.column_index, 
        att, 
        inputInfo.t_window_rowTensor,
        inputInfo.t_atomicTensor,
        X_prime, 
        inputInfo.num_nodes, 
        X_prime.size(1), 
        inputInfo.num_nodes_ori)[0]
        return X_prime.half()

    @staticmethod
    def backward(ctx, d_output):
        inputInfo = ctx.inputInfo
        X_prime = ctx.X_prime
        att = ctx.att
        
        # print(d_output.shape())
        # print(X_prime.shape())
        
        #SDMM 求att的梯度
        d_attention = FS_SDDMM.forward_gen_fp16_gnn(   
            X_prime.size(1),                                      
            inputInfo.row_pointers, 
            inputInfo.column_index, 
            inputInfo.degrees, 
            inputInfo.t_window_rowTensor,
            d_output,X_prime,inputInfo.max)[0]  
        
        # SPMM backward propaAGNNion.
        d_input_prime  = FS_SpMM.forward_fp16_gnn(   
        inputInfo.row_pointers, 
        inputInfo.column_index, 
        att, 
        inputInfo.t_window_rowTensor,
        inputInfo.t_atomicTensor,
        d_output, 
        inputInfo.num_nodes, 
        d_output.size(1), 
        inputInfo.num_nodes_ori)[0]

        
        return d_attention, d_input_prime, None
    
class MAGNNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_prime, attention_w, inputInfo):

        edge_feature = FS_SDDMM.forward_gen_fp16_gnn(   
            X_prime.size(1),                                      
            inputInfo.row_pointers, 
            inputInfo.column_index, 
            inputInfo.degrees, 
            inputInfo.t_window_rowTensor,
            X_prime,X_prime,inputInfo.max)[0]                   
                                                  
        edge_feature = edge_feature * attention_w
       
        return edge_feature
    @staticmethod
    def backward(ctx, d_attention):
        d_attention_w = torch.sum(d_attention).view(1)
        return None, d_attention_w, None


class MAGNNSpmm1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, att, X_prime, inputInfo):
        # SpMM: Neighbor AggreAGNNion.
        X_prime  = FS_SpMM.forward_fp16_gnn_ones(   
            inputInfo.row_pointers, 
            inputInfo.column_index, 
            att, 
            inputInfo.t_window_rowTensor,
            inputInfo.t_atomicTensor,
            X_prime, 
            inputInfo.num_nodes, 
            X_prime.size(1), 
            inputInfo.num_nodes_ori)[0]
        return X_prime
    @staticmethod
    def backward(ctx, X_prime_grad):
        return None, None, None
   
class AGNNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AGNNConv, self).__init__()
        # gain1 = nn.init.calculate_gain("relu")
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))

        self.attention_w = torch.nn.Parameter(torch.randn(1))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

        
    def forward(self, X, inputInfo):
        # 1. 特征降维
        X_prime = torch.mm(X, self.weights.half())
        
        # 2. 求att
        att = MAGNNFunction.apply(X_prime, self.attention_w.half(), inputInfo)
        print(att.shape)
        
        # # 3. exp
        # max_value= torch.max(att)
        # min_value= torch.min(att)
        # att = (att - min_value) / (max_value - min_value)
        #temp = att
        att = torch.exp(att)
        rows_sum = MAGNNSpmm1.apply(att, inputInfo.ones, inputInfo)

        # 4. 特征更新
        h_prime = MAGNNSpmm.apply(att, X_prime, inputInfo)
        
        # # 5. softmax
        h_prime = h_prime.div(rows_sum)

        return h_prime.half()