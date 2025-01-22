import os.path as osp
import argparse
import time
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
from agnn_dataset import *
from magnn_conv import *
from torch.optim import Adam


#########################################
## Build GCN and AGNN Model
#########################################

class Net(torch.nn.Module):
    def __init__(self,in_size, hid_size, out_size, num_layers):
        super(Net, self).__init__()
        self.lin1 = torch.nn.Linear(in_size, hid_size).half()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(AGNNConv(hid_size, hid_size ))
        self.lin2 = torch.nn.Linear(hid_size, out_size).half()


    def forward(self, inputInfo):
        h = inputInfo.x
        h = F.relu(self.lin1(h))
        for conv in self.convs:
            h = F.relu(conv(h, inputInfo))
        h = self.lin2(h)
        return F.log_softmax(h, dim=1)
    
# Training 
def train(model, inputInfo, epoches):
    # loss_fcn = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)  
    
    for epoch in range(epoches):
        model.train()
        logits = model(inputInfo)
        # if torch.isnan(logits).any().item() :
        loss =  F.nll_loss(logits, inputInfo.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()