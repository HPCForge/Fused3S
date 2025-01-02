import torch
from DFGNN.layers.util import preprocess_softmax
from dgl.data import (CoraGraphDataset, RedditDataset, PubmedGraphDataset)
import TCFMM
import argparse


def run_f3s(indptr, indices, num_nodes, num_edges, Q, K, V, BLK_H, BLK_W, n_warp_per_block):
  # Set up tensors for preprocessing
  num_row_windows = (num_nodes + BLK_H - 1) // BLK_H
  # Move tensors to GPU
  blockPartition_cuda = torch.zeros(num_row_windows, dtype=torch.int).cuda()
  edgeToColumn_cuda = torch.zeros(num_edges, dtype=torch.int).cuda()
  edgeToRow_cuda = torch.zeros(num_edges, dtype=torch.int).cuda()
  RowWindowOffset, TCblockRowid, TCblocktileId,\
  TCblockoffset, SparseAToXindex, TBBoundaries, TCblockBitMap,\
  block_count = TCFMM.preprocess_gpu(indices, indptr, num_nodes, BLK_H, BLK_W, blockPartition_cuda, edgeToColumn_cuda, edgeToRow_cuda)
  apply_softmax = True
  save_sddmm_result = True
  sddmm_result_1tb1tcb = TCFMM.f3s_1tb1tcb(RowWindowOffset, SparseAToXindex, TCblockBitMap, num_nodes, Q, K, V, apply_softmax, save_sddmm_result)[1]
  sddmm_result_1tb1rw = TCFMM.f3s_1tb1rw(RowWindowOffset, SparseAToXindex, TCblockBitMap, num_nodes, Q, K, V, n_warp_per_block)[1]
  #last true/false is for use_1tb1rw, true is 1tb1rw
  sddmm_result_sddmm_1tb1rw = TCFMM.f3S_sddmm(RowWindowOffset, TBBoundaries, TCblockRowid, SparseAToXindex, TCblockBitMap, num_nodes, Q, K, n_warp_per_block, True)[0]
  sddmm_result_sddmm_1tbnrw = TCFMM.f3S_sddmm(RowWindowOffset, TBBoundaries, TCblockRowid, SparseAToXindex, TCblockBitMap, num_nodes, Q, K, n_warp_per_block, False)[0]
  # return sddmm_result_forward, sddmm_result_sddmm
  
def main(embedding_dim, n_warp_per_block):
  BLK_H = 16
  BLK_W = 8
  num_heads = 1
  dataset_dir = "/share/crsp/lab/amowli/share/Fused3S/dfgnn/"
  dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  dataset = RedditDataset(raw_dir=dataset_dir)
  g = dataset[0].to(dev)
  num_nodes = g.num_nodes()
  num_edges = g.num_edges()
  print(g)
  print(g.ndata["feat"].shape)
  params = preprocess_softmax(g)
  del g  
  torch.cuda.empty_cache()
  indptr, indices, rows, val, smem_consume = params
  Q = torch.rand(num_nodes, embedding_dim, dtype=torch.float16, device=dev)
  K = torch.rand(num_nodes, embedding_dim, dtype=torch.float16, device=dev)
  V = torch.rand(num_nodes, embedding_dim, dtype=torch.float16, device=dev)

  run_f3s(indptr, indices, num_nodes, num_edges, Q, K, V, BLK_H, BLK_W, n_warp_per_block)
  

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', '-emb', type=int, default=320,
                       help='Dimension of node embeddings')
    parser.add_argument('--n_warp_per_block', '-nw', type=int, default=16,
                       help='Number of warps per block')
    args = parser.parse_args()
    embedding_dim = args.embedding_dim
    n_warp_per_block = args.n_warp_per_block
    main(embedding_dim, n_warp_per_block)
