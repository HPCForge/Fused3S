import torch
from DFGNN.layers.util import preprocess_softmax
from dgl.data import (CoraGraphDataset, RedditDataset, PubmedGraphDataset)
import TCFMM
import argparse
from ogb.linkproppred import DglLinkPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset
import dgl.sparse as dglsp

def preprocess_dglsp(g):
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))
    return A

def run_f3s_1tb1rw(indptr, indices, num_nodes, num_edges, Q, K, V, n_warp_per_block):
  BLK_H = 16
  BLK_W = 16
  # Set up tensors for preprocessing
  num_row_windows = (num_nodes + BLK_H - 1) // BLK_H
  # Move tensors to GPU
  blockPartition_cuda = torch.zeros(num_row_windows, dtype=torch.int).cuda()
  edgeToColumn_cuda = torch.zeros(num_edges, dtype=torch.int).cuda()
  edgeToRow_cuda = torch.zeros(num_edges, dtype=torch.int).cuda()
  RowWindowOffset, TCblockRowid, TCblocktileId,\
  TCblockoffset, SparseAToXindex, TBBoundaries, TCblockBitMap,\
  block_count = TCFMM.preprocess_gpu(indices, indptr, num_nodes, BLK_H, BLK_W, blockPartition_cuda, edgeToColumn_cuda, edgeToRow_cuda)
  print("preprocess done")
  sddmm_result_1tb1rw = TCFMM.f3s_1tb1rw(RowWindowOffset, SparseAToXindex, TCblockBitMap, num_nodes, Q, K, V, n_warp_per_block)[1]

def run_f3s_1tb1tcb(indptr, indices, num_nodes, num_edges, Q, K, V):
  BLK_H = 16
  BLK_W = 8
  apply_softmax = True
  save_sddmm_result = False
  # Set up tensors for preprocessing
  num_row_windows = (num_nodes + BLK_H - 1) // BLK_H
  # Move tensors to GPU
  blockPartition_cuda = torch.zeros(num_row_windows, dtype=torch.int).cuda()
  edgeToColumn_cuda = torch.zeros(num_edges, dtype=torch.int).cuda()
  edgeToRow_cuda = torch.zeros(num_edges, dtype=torch.int).cuda()
  RowWindowOffset, TCblockRowid, TCblocktileId,\
  TCblockoffset, SparseAToXindex, TBBoundaries, TCblockBitMap,\
  block_count = TCFMM.preprocess_gpu(indices, indptr, num_nodes, BLK_H, BLK_W, blockPartition_cuda, edgeToColumn_cuda, edgeToRow_cuda)
  sddmm_result_1tb1tcb = TCFMM.f3s_1tb1tcb(RowWindowOffset, SparseAToXindex, TCblockBitMap, num_nodes, Q, K, V, apply_softmax, save_sddmm_result)[1]
  
def main(args):
  BLK_H = 16
  BLK_W = 8
  num_heads = 1
  dataset_dir = "/share/crsp/lab/amowli/share/Fused3S/dfgnn/"
  dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  if args.dataset == "reddit":
    dataset = RedditDataset(raw_dir=dataset_dir)
  elif args.dataset == "ppa":
    dataset = DglLinkPropPredDataset(name="ogbl-ppa", root=dataset_dir)
  elif args.dataset == "protein":
    dataset = DglNodePropPredDataset(name="ogbn-proteins", root=dataset_dir)
  elif args.dataset == "cora":
    dataset = CoraGraphDataset(raw_dir=dataset_dir)
  elif args.dataset == "pubmed":
    dataset = PubmedGraphDataset(raw_dir=dataset_dir)
  else:
    raise ValueError(f"dataset: {dataset} not supported in this test")
  g = dataset[0].to(dev)
  indptr, indices, vals = g.adj().csr()
  indptr = indptr.to(torch.int32)
  indices = indices.to(torch.int32)
  num_nodes = g.num_nodes()
  num_edges = g.num_edges()

  Q = torch.rand(num_nodes, args.embedding_dim, dtype=torch.float16, device=dev)
  K = torch.rand(num_nodes, args.embedding_dim, dtype=torch.float16, device=dev)
  V = torch.rand(num_nodes, args.embedding_dim, dtype=torch.float16, device=dev)

  if args.n_warp_per_block == -1:
    for n_warp_per_block in [4, 8, 16]:
      run_f3s_1tb1rw(indptr, indices, num_nodes, num_edges, Q, K, V, n_warp_per_block)
  else:
    run_f3s_1tb1rw(indptr, indices, num_nodes, num_edges, Q, K, V, args.n_warp_per_block)
  run_f3s_1tb1tcb(indptr, indices, num_nodes, num_edges, Q, K, V)
  

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', '-emb', type=int, default=128,
                       help='Dimension of node embeddings')
    parser.add_argument('--n_warp_per_block', '-nw', type=int, default=-1,
                       help='Number of warps per block')
    parser.add_argument('--dataset', '-d', type=str, default="reddit",
                       choices=["reddit", "ppa", "protein", "cora", "pubmed"])
    args = parser.parse_args()
    main(args)
