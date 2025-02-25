import torch
from DFGNN.layers import load_prepfunc
from DFGNN.layers.util import preprocess_softmax, preprocess_CSR
from DFGNN.operators.fused_gtconv import GTConvFuse_inference_softmax, GTConvFuse_inference_csr, GTConvFuse_inference_hyper
from dgl.data import (CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, RedditDataset)
import argparse
import TCFMM

def run_f3s_1tb1rw(indptr, indices, num_nodes, num_edges, Q, K, V, n_warp_per_block):
  BLK_H = 16
  BLK_W = 16
  # Set up tensors for preprocessing
  num_row_windows = (num_nodes + BLK_H - 1) // BLK_H
  # Move tensors to GPU
  blockPartition_cuda = torch.zeros(num_row_windows, dtype=torch.int).cuda()
  edgeToColumn_cuda = torch.zeros(num_edges, dtype=torch.int).cuda()
  edgeToRow_cuda = torch.zeros(num_edges, dtype=torch.int).cuda()
  RowWindowOffset, sortedRowWindows, TCblockRowid, TCblocktileId,\
  TCblockoffset, SparseAToXindex, TBBoundaries, TCblockBitMap,\
  block_count = TCFMM.preprocess_gpu(indices, indptr, num_nodes, BLK_H, BLK_W, blockPartition_cuda, edgeToColumn_cuda, edgeToRow_cuda)
  print("preprocess done")
  result_1tb1rw, sddmm_result = TCFMM.f3s_1tb1rw(RowWindowOffset, SparseAToXindex, TCblockBitMap, num_nodes, Q, K, V, n_warp_per_block)
  torch.cuda.synchronize()
  return result_1tb1rw

def main(args):
  num_heads = 1
  dataset_dir = "/share/crsp/lab/amowli/share/Fused3S/dfgnn/"
  dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  if args.dataset == "citeseer":
    dataset = CiteseerGraphDataset(raw_dir=dataset_dir)
  elif args.dataset == "pubmed":
    dataset = PubmedGraphDataset(raw_dir=dataset_dir)
  elif args.dataset == "cora":
    dataset = CoraGraphDataset(raw_dir=dataset_dir)
  elif args.dataset == "reddit":
    dataset = RedditDataset(raw_dir=dataset_dir)
  else:
    raise ValueError(f"Invalid dataset: {args.dataset}")

  g = dataset[0].to(dev)
  num_nodes = g.num_nodes()
  num_edges = g.num_edges()
  Q = torch.rand(num_nodes, num_heads, args.embedding_dim, dtype=torch.float32, device=dev)
  K = torch.rand(num_nodes, num_heads, args.embedding_dim, dtype=torch.float32, device=dev)
  V = torch.rand(num_nodes, num_heads, args.embedding_dim, dtype=torch.float32, device=dev)
  print(g.ndata["feat"].shape)
  preprocess_func = load_prepfunc(args)
  if args.format == "softmax":
    indptr, indices, rows, val, smem_consume = preprocess_func(g)
    out = GTConvFuse_inference_softmax(indptr, indices, rows, val, smem_consume, Q, K, V)
  elif args.format == "csr":
    indptr, indices, val, smem_consume = preprocess_func(g)
    out = GTConvFuse_inference_csr(indptr, indices, val, smem_consume, Q, K, V)
  elif args.format == "hyper":
    indptr, indices, rows, val, smem_consume = preprocess_func(g)
    out = GTConvFuse_inference_hyper(indptr, indices, rows, val, smem_consume, Q, K, V)
  else:
    raise ValueError(f"Invalid format: {args.format}")
  torch.cuda.synchronize()
  # get rid of head dimension
  out = out.view(num_nodes, args.embedding_dim)
  print(out.shape)

  n_warp_per_block = 8
  Q_half = Q.view(num_nodes, args.embedding_dim).half()
  K_half = K.view(num_nodes, args.embedding_dim).half()
  V_half = V.view(num_nodes, args.embedding_dim).half()
  result_1tb1rw = run_f3s_1tb1rw(indptr, indices, num_nodes, num_edges, Q_half, K_half, V_half, n_warp_per_block)
  print(result_1tb1rw.shape)
  rel_error = torch.norm(out - result_1tb1rw)/torch.norm(out)
  print(f"relative error: {rel_error}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', '-emb', type=int, default=128,
                       help='Dimension of node embeddings')
    parser.add_argument('--format', '-f', type=str, default="softmax",
                        choices=["softmax", "csr", "hyper"],
                        help='Format to use')
    parser.add_argument('--dataset', '-d', type=str, default="pubmed",
                       choices=["citeseer", "pubmed", "cora", "reddit"],
                       help='Dataset to use')
    args = parser.parse_args()
    main(args)
