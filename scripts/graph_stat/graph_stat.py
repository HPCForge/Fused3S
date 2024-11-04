import os.path as osp
import pandas as pd
import torch
import matplotlib.pyplot as plt
import sys
BLK_H = 8
BLK_W = 8
import DTCSpMM
sys.path.append('/pub/zitongl5/FTC-MM/scripts/DTCSpMM')
from dataset import *
from plot_grid import *

plot_graph_stat = False

def visualize_sparsity(sparse_matrix, title="Sparsity Pattern"):
    plt.figure(figsize=(10, 10))
    plt.spy(sparse_matrix, markersize=1)
    plt.title(title)
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    
    # Show the plot
    plt.show()
    plt.savefig(title + ".png")

graph_reorder = ["citeseer.reorder", "cora.reorder", "YeastH.reorder", "OVCAR-8H.reorder", "Yeast.reorder", "DD.reorder", "soc-BlogCatalog.reorder", "web-BerkStan.reorder", "reddit.reorder", "ddi.reorder", "protein.reorder"]
graph_og = ["citeseer", "cora", "YeastH", "OVCAR-8H", "Yeast", "DD", "soc-BlogCatalog", "web-BerkStan", "reddit", "ddi", "protein"]
# graph_reorder = ["YeastH.reorder", "OVCAR-8H.reorder"]
# graph_og = ["YeastH", "OVCAR-8H"]
num_graph = len(graph_reorder)
df_reorder = np.zeros([num_graph, 14])
df_og = np.zeros([num_graph, 14])
pg = PlotGrid(5, max(num_graph, 2))
for graph_set in [graph_og, graph_reorder]:
  df = df_reorder if graph_set == graph_reorder else df_og
  ind = 0 if graph_set == graph_reorder else 1
  title_prefix = "RO" if graph_set == graph_reorder else "OG"
  nBlockPerRowWindow_lst = []
  nnzPerBlock_lst = []
  for i, dataset in enumerate(graph_set):
    # Set your own path to the dataset.
    path = osp.join("/pub/zitongl5/DTC-SpMM-Datasets/", dataset + ".npz") 
    matrix = DTC_dataset(path)
    num_rows = matrix.num_nodes
    num_nnz = matrix.num_edges
    print("dataset: ", dataset, "NUM_ROW, NNZ: ", num_rows, " " , num_nnz)
    df[i, 0] = num_rows
    df[i, 1] = num_nnz
    df[i, 2] = num_nnz / (num_rows * num_rows)
    # Process data.
    column_index =  matrix.column_index 
    row_pointers =  matrix.row_pointers 
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
    df[i, 3] = block_count

    RowWindowOffset = RowWindowOffset.cpu().numpy()
    TCblockoffset = TCblockoffset.cpu().numpy()
    nBlockPerRowWindow = np.array([j - i for i, j in zip(RowWindowOffset[:-1], RowWindowOffset[1:])])
    nnzPerBlock = np.array([j - i for i, j in zip(TCblockoffset[:-1], TCblockoffset[1:])])

    # Check nnzPerBlock
    if matrix.A_csr.indptr[50*BLK_H] - matrix.A_csr.indptr[0] != sum(nnzPerBlock[:sum(nBlockPerRowWindow[:50])]):
      print("nnz in first 50 row windows: ", sum(nnzPerBlock[:sum(nBlockPerRowWindow[:50])]))
      print("true nnz in first 50 row windows: ", matrix.A_csr.indptr[50*BLK_H] - matrix.A_csr.indptr[0])
      raise Exception("nnz in first 50 row windows is incorrect")
    
    nBlockPerRowWindow_lst.append(nBlockPerRowWindow)
    nnzPerBlock_lst.append(nnzPerBlock)
    avg_nnz_per_block = sum(nnzPerBlock) / len(nnzPerBlock)
    avg_block_per_row_window = sum(nBlockPerRowWindow) / len(nBlockPerRowWindow)
    variance_nnz_per_block = sum([(x - avg_nnz_per_block) ** 2 for x in nnzPerBlock]) / len(nnzPerBlock)
    variance_block_per_row_window = sum([(x - avg_block_per_row_window) ** 2 for x in nBlockPerRowWindow]) / len(nBlockPerRowWindow)  
    median_nnz_per_block = np.median(nnzPerBlock)
    first_quartile_nnz_per_block = np.percentile(nnzPerBlock, 25)
    third_quartile_nnz_per_block = np.percentile(nnzPerBlock, 75)
    median_block_per_row_window = np.median(nBlockPerRowWindow)
    first_quartile_block_per_row_window = np.percentile(nBlockPerRowWindow, 25)
    third_quartile_block_per_row_window = np.percentile(nBlockPerRowWindow, 75)
    print("avg_nnz_per_block: ", avg_nnz_per_block)
    print("variance_nnz_per_block: ", variance_nnz_per_block)
    print("median_nnz_per_block: ", median_nnz_per_block)
    print("first_quartile_nnz_per_block: ", first_quartile_nnz_per_block)
    print("third_quartile_nnz_per_block: ", third_quartile_nnz_per_block)
    print("avg_block_per_row_window: ", avg_block_per_row_window)
    print("variance_block_per_row_window: ", variance_block_per_row_window)
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
    # only plot the title in the first row
    if plot_graph_stat:
      if graph_set == graph_og:
        pg.make_text_sub_plot(f"num_rows: {num_rows}\nnnz_ratio: {df[i, 2]:.2e}", dataset, [0, i])
      pg.make_spy_plot(matrix.A_csr[0:1000, 0:1000], title_prefix + "_A[:1k, :1k]", [ind*2+1, i])
      pg.make_spy_plot(matrix.A_csr, title_prefix + "_full A", [ind*2+2, i])
    print("================================================================")

  # Pad nBlockPerRowWindow to be a regular matrix
  max_length = max(len(arr) for arr in nBlockPerRowWindow_lst)
  padded_nBlockPerRowWindow_lst = [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=0) for arr in nBlockPerRowWindow_lst]
  nBlockPerRowWindow_matrix = np.array(padded_nBlockPerRowWindow_lst).T

  # Pad nnzPerBlock to be a regular matrix
  max_length = max(len(arr) for arr in nnzPerBlock_lst)
  padded_nnzPerBlock_lst = [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=0) for arr in nnzPerBlock_lst]
  nnzPerBlock_matrix = np.array(padded_nnzPerBlock_lst).T

  nBlockPerRowWindow_pd = pd.DataFrame(nBlockPerRowWindow_matrix, columns=graph_set)
  nBlockPerRowWindow_pd.to_csv(title_prefix + "_" + str(BLK_H) + "x" + str(BLK_W) + "_nBlockPerRowWindow_pd.csv")
  nnzPerBlock_pd = pd.DataFrame(nnzPerBlock_matrix, columns=graph_set)
  nnzPerBlock_pd.to_csv(title_prefix + "_" + str(BLK_H) + "x" + str(BLK_W) + "_nnzPerBlock_pd.csv")


columns = ["graph", "num_rows", "num_nnz", "nnz_percentage", "block_count", "avg_nnz_per_block", "variance_nnz_per_block", "median_nnz_per_block", "first_quartile_nnz_per_block", "third_quartile_nnz_per_block", "avg_block_per_row_window", "variance_block_per_row_window", "median_block_per_row_window", "first_quartile_block_per_row_window", "third_quartile_block_per_row_window"]
df_reorder = np.concatenate([np.array([graph_reorder]).T, df_reorder], axis=1)
df_reorder = pd.DataFrame(df_reorder, columns=columns)
df_reorder.to_csv(str(BLK_H) + "x" + str(BLK_W) + "_graph_stat_reorder.csv", index=False)
df_og = np.concatenate([np.array([graph_og]).T, df_og], axis=1)
df_og = pd.DataFrame(df_og, columns=columns)
df_og.to_csv(str(BLK_H) + "x" + str(BLK_W) + "_graph_stat_og.csv", index=False)
if plot_graph_stat:
  pg.save("graph_stat.png")

