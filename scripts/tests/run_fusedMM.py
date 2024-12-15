import os.path as osp
import argparse
import torch

BLK_H = 16
BLK_W = 16
import DTCSpMM
from dataset import *

parser = argparse.ArgumentParser()
parser.add_argument("--feat_size", type=int, default=128, help="embedding size")
parser.add_argument("--dataset", type=str, default='YeastH', help="dataset")
parser.add_argument("--save_edge_attention", type=bool, default=False, help="save edge attention")
args = parser.parse_args()
print(args)

## Load matrix from files.
dataset = args.dataset
dset_name = dataset
# Set your own path to the dataset.
path = osp.join("/pub/zitongl5/DTC-SpMM-Datasets/", dataset + ".npz") #4090
matrix = DTC_dataset(path)
num_rows = matrix.num_nodes
num_nnz = matrix.num_edges
print("NUM_ROW, NNZ: ", num_rows, " " , num_nnz)
column_index =  matrix.column_index 
row_pointers =  matrix.row_pointers 
# Process data.
num_row_windows = (num_rows + BLK_H - 1) // BLK_H
edgeToColumn = torch.zeros(num_nnz, dtype=torch.int)
edgeToRow = torch.zeros(num_nnz, dtype=torch.int)
blockPartition = torch.zeros(num_row_windows, dtype=torch.int)
column_index_ori  = column_index.cuda()
row_pointers_ori = row_pointers.cuda()

blockPartition_cuda  = blockPartition.cuda()
edgeToColumn_cuda = edgeToColumn.cuda()
edgeToRow_cuda  = edgeToRow.cuda()

# Optimize GPU.
RowWindowOffset, TCblockRowid,\
      TCblocktileId, TCblockoffset, SparseAToXindex,\
        block_count = DTCSpMM.preprocess_gpu(column_index_ori, row_pointers_ori, num_rows, BLK_H, BLK_W, blockPartition_cuda, edgeToColumn_cuda, edgeToRow_cuda)
# Run tests.
print("feat_size =", args.feat_size)
X = torch.ones((num_rows, args.feat_size)).cuda()
# Run test.
use_f32_edge_attention = True
fusedR, edgeAttention = DTCSpMM.fusedMM_forward(X.to(torch.float16), RowWindowOffset, TCblocktileId, TCblockoffset, SparseAToXindex, num_rows, args.save_edge_attention, use_f32_edge_attention)
