import os.path as osp
import pandas as pd
import torch
import matplotlib.pyplot as plt
import sys
BLK_H = 16
BLK_W = 8
import numpy as np
from scipy.sparse import coo_matrix
from F3S import preprocess_gpu

datasets = ["reddit", "amazonProducts", "yelp", "amazon0505", 
            "Artist", "Blog", "com-amazon.ungraph", "github", 
            "Ell", "ogbn-products", "citeseer", "pubmed", "cora",
            "igb_small", "igb_medium"]
columns = ["graph", "num_rows", "num_nnz", "nnz_percentage", "block_count", "avg_nnz_per_block", "variance_nnz_per_block", "median_nnz_per_block", "first_quartile_nnz_per_block", "third_quartile_nnz_per_block", "avg_block_per_row_window", "variance_block_per_row_window", "median_block_per_row_window", "first_quartile_block_per_row_window", "third_quartile_block_per_row_window"]

num_graph = len(datasets)
df = np.zeros([num_graph, len(columns)])

def load_dataset(dataset_name):
  print(f"===========loading dataset: {dataset_name}===========")
  # Check available GPU memory before loading dataset
  path = f"/share/crsp/lab/amowli/share/Fused3S/dataset/{dataset_name}.npz"
  dataset = np.load(path)
  src_li = dataset['src_li'] # this can contain duplicate edges
  dst_li = dataset['dst_li']
  val = [1] * len(src_li)
  edge_index = np.stack([src_li, dst_li])
  scipy_coo = coo_matrix((val, edge_index), shape=(dataset['num_nodes'], dataset['num_nodes']))
  adj = scipy_coo.tocsr()
  print(f"dataset: {dataset_name}, num_nodes: {adj.shape[0]}, num_edges: {adj.nnz}")
  return adj

nBlockPerRowWindow_lst = []
nnzPerBlock_lst = []
for i, dataset in enumerate(datasets):
  # Set your own path to the dataset.
  matrix = load_dataset(dataset)
  num_rows = matrix.shape[0]
  num_nnz = matrix.nnz
  df[i, 0] = num_rows
  df[i, 1] = num_nnz
  df[i, 2] = num_nnz / (num_rows * num_rows)
  # Process data.
  row_pointers =  torch.IntTensor(matrix.indptr).cuda()
  column_index =  torch.IntTensor(matrix.indices).cuda()
  num_row_windows = (num_rows + BLK_H - 1) // BLK_H
  edgeToColumn = torch.zeros(num_nnz, dtype=torch.int)
  edgeToRow = torch.zeros(num_nnz, dtype=torch.int)
  blockPartition = torch.zeros(num_row_windows, dtype=torch.int)

  blockPartition_cuda  = blockPartition.cuda()
  edgeToColumn_cuda = edgeToColumn.cuda()
  edgeToRow_cuda  = edgeToRow.cuda()

  # Optimize GPU.
  RowWindowOffset, sortedRowWindows, TCblockRowid,\
  TCblocktileId, TCblockoffset, SparseAToXindex,\
  TCblockBitMap, block_count = preprocess_gpu(column_index, row_pointers, num_rows, BLK_H, BLK_W, blockPartition_cuda, edgeToColumn_cuda, edgeToRow_cuda)
  df[i, 3] = block_count

  RowWindowOffset = RowWindowOffset.cpu().numpy()
  TCblockoffset = TCblockoffset.cpu().numpy()
  nBlockPerRowWindow = np.array([j - i for i, j in zip(RowWindowOffset[:-1], RowWindowOffset[1:])])
  nnzPerBlock = np.array([j - i for i, j in zip(TCblockoffset[:-1], TCblockoffset[1:])])

  # Check nnzPerBlock
  if matrix.indptr[50*BLK_H] - matrix.indptr[0] != sum(nnzPerBlock[:sum(nBlockPerRowWindow[:50])]):
    print("nnz in first 50 row windows: ", sum(nnzPerBlock[:sum(nBlockPerRowWindow[:50])]))
    print("true nnz in first 50 row windows: ", matrix.indptr[50*BLK_H] - matrix.indptr[0])
    raise Exception("nnz in first 50 row windows is incorrect")
  
  nBlockPerRowWindow_lst.append(nBlockPerRowWindow)
  nnzPerBlock_lst.append(nnzPerBlock)
  avg_nnz_per_block = sum(nnzPerBlock) / len(nnzPerBlock)
  avg_block_per_row_window = sum(nBlockPerRowWindow) / len(nBlockPerRowWindow)
  variance_nnz_per_block = sum([(x - avg_nnz_per_block) ** 2 for x in nnzPerBlock]) / len(nnzPerBlock)
  variance_block_per_row_window = sum([(x - avg_block_per_row_window) ** 2 for x in nBlockPerRowWindow]) / len(nBlockPerRowWindow)  
  median_nnz_per_block = np.median(nnzPerBlock)
  std_nnz_per_block = np.std(nnzPerBlock)
  rsd_nnz_per_block = std_nnz_per_block / avg_nnz_per_block
  first_quartile_nnz_per_block = np.percentile(nnzPerBlock, 25)
  third_quartile_nnz_per_block = np.percentile(nnzPerBlock, 75)
  median_block_per_row_window = np.median(nBlockPerRowWindow)
  std_block_per_row_window = np.std(nBlockPerRowWindow)
  rsd_block_per_row_window = std_block_per_row_window / avg_block_per_row_window
  first_quartile_block_per_row_window = np.percentile(nBlockPerRowWindow, 25)
  third_quartile_block_per_row_window = np.percentile(nBlockPerRowWindow, 75)
  print("avg_nnz_per_block: ", avg_nnz_per_block)
  print("variance_nnz_per_block: ", variance_nnz_per_block)
  print("median_nnz_per_block: ", median_nnz_per_block)
  print("rsd_nnz_per_block: ", rsd_nnz_per_block)
  print("first_quartile_nnz_per_block: ", first_quartile_nnz_per_block)
  print("third_quartile_nnz_per_block: ", third_quartile_nnz_per_block)
  print("")
  print("avg_block_per_row_window: ", avg_block_per_row_window)
  print("variance_block_per_row_window: ", variance_block_per_row_window)
  print("rsd_block_per_row_window: ", rsd_block_per_row_window)
  print("median_block_per_row_window: ", median_block_per_row_window)
  print("first_quartile_block_per_row_window: ", first_quartile_block_per_row_window)
  print("third_quartile_block_per_row_window: ", third_quartile_block_per_row_window)
  df[i, 4] = avg_nnz_per_block
  df[i, 5] = variance_nnz_per_block
  df[i, 6] = median_nnz_per_block
  df[i, 7] = first_quartile_nnz_per_block
  df[i, 8] = third_quartile_nnz_per_block
  df[i, 9] = avg_block_per_row_window
  df[i, 10] = variance_block_per_row_window
  df[i, 11] = median_block_per_row_window
  df[i, 12] = first_quartile_block_per_row_window
  df[i, 13] = third_quartile_block_per_row_window

# Pad nBlockPerRowWindow to be a regular matrix
max_length = max(len(arr) for arr in nBlockPerRowWindow_lst)
padded_nBlockPerRowWindow_lst = [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=0) for arr in nBlockPerRowWindow_lst]
nBlockPerRowWindow_matrix = np.array(padded_nBlockPerRowWindow_lst).T

# Pad nnzPerBlock to be a regular matrix
max_length = max(len(arr) for arr in nnzPerBlock_lst)
padded_nnzPerBlock_lst = [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=0) for arr in nnzPerBlock_lst]
nnzPerBlock_matrix = np.array(padded_nnzPerBlock_lst).T

nBlockPerRowWindow_pd = pd.DataFrame(nBlockPerRowWindow_matrix, columns=datasets)
nBlockPerRowWindow_pd.to_csv("paper_no_reorder_" + str(BLK_H) + "x" + str(BLK_W) + "_nBlockPerRowWindow_pd.csv")
nnzPerBlock_pd = pd.DataFrame(nnzPerBlock_matrix, columns=datasets)
nnzPerBlock_pd.to_csv("paper_no_reorder_" + str(BLK_H) + "x" + str(BLK_W) + "_nnzPerBlock_pd.csv")
