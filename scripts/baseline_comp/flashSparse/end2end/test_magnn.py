import os.path as osp
import argparse
import time
import torch
import sys
from agnn_dataset import *
from magnn_conv import *
from agnn_mgnn import *


def test(data, epoches, layers, featuredim, hidden, classes):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    inputInfo = MAGNN_dataset(data, featuredim , classes)

    inputInfo.to(device)
    model= Net(inputInfo.num_features, hidden, inputInfo.num_classes, layers).to(device)

    train(model, inputInfo, 1)
    torch.cuda.synchronize()
    start_time = time.time()
    train(model, inputInfo, epoches)
    torch.cuda.synchronize()
    end_time = time.time()
    execution_time = end_time - start_time
    print("execution_time: ", execution_time)

if __name__ == "__main__":

    test('/pub/zitongl5/DTC-SpMM-Datasets/reddit.npz', epoches=1, layers=1, featuredim=3, hidden=128, classes=10)