import torch
from DFGNN.layers.util import preprocess_softmax
from dgl.data import (CoraGraphDataset, RedditDataset, PubmedGraphDataset)
import TCFMM
import argparse
from ogb.linkproppred import DglLinkPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset

def timing_decorator(kernel):
  def wrapper(*args, **kwargs):
    # warmup
    for i in range(10):
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      start_event.record()
      out = kernel(*args, **kwargs)
      end_event.record()
      torch.cuda.synchronize()
      execution_time = start_event.elapsed_time(end_event)
      print(f"execution time: {execution_time} ms")
    return out
  return wrapper

def main(args):
  BLK_H = 16
  if args.alg == '1tb1rw' or args.alg == '1tb1rw_scheduled':
    BLK_W = 16
  else:
    BLK_W = 8
  num_heads = 1
  dataset_dir = "/share/crsp/lab/amowli/share/Fused3S/dataLoader/"
  dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  if args.dataset == "reddit":
    dataset = RedditDataset(raw_dir=dataset_dir+args.dataset)
  elif args.dataset == "ppa":
    dataset = DglLinkPropPredDataset(name="ogbl-ppa", root=dataset_dir+args.dataset)
  elif args.dataset == "protein":
    dataset = DglNodePropPredDataset(name="ogbn-proteins", root=dataset_dir+args.dataset)
  elif args.dataset == "cora":
    dataset = CoraGraphDataset(raw_dir=dataset_dir+args.dataset)
  elif args.dataset == "pubmed":
    dataset = PubmedGraphDataset(raw_dir=dataset_dir+args.dataset)
  else:
    raise ValueError(f"dataset: {dataset} not supported in this test")
  g = dataset[0].to(dev)
  indptr, indices, vals = g.adj().csr()
  indptr = indptr.to(torch.int32)
  indices = indices.to(torch.int32)
  num_nodes = g.num_nodes()
  num_edges = g.num_edges()
  print(f"dataset: {args.dataset}, num_nodes: {num_nodes}, num_edges: {num_edges}")

  Q = torch.rand(num_nodes, args.embedding_dim, dtype=torch.float16, device=dev)
  K = torch.rand(num_nodes, args.embedding_dim, dtype=torch.float16, device=dev)
  V = torch.rand(num_nodes, args.embedding_dim, dtype=torch.float16, device=dev)

  # Set up tensors for preprocessing
  num_row_windows = (num_nodes + BLK_H - 1) // BLK_H
  # Move tensors to GPU
  blockPartition_cuda = torch.zeros(num_row_windows, dtype=torch.int).cuda()
  edgeToColumn_cuda = torch.zeros(num_edges, dtype=torch.int).cuda()
  edgeToRow_cuda = torch.zeros(num_edges, dtype=torch.int).cuda()
  RowWindowOffset, sortedRowWindows, TCblockRowid, TCblocktileId,\
  TCblockoffset, SparseAToXindex, TBBoundaries, TCblockBitMap,\
  block_count = TCFMM.preprocess_gpu(indices, indptr, num_nodes, 
                                     BLK_H, BLK_W, blockPartition_cuda, 
                                     edgeToColumn_cuda, edgeToRow_cuda)

  if args.use_cuda_event:
    f3s_1tb1rw = timing_decorator(TCFMM.f3s_1tb1rw)
    f3s_1tb1rw_scheduled= timing_decorator(TCFMM.f3s_1tb1rw_scheduled)
    f3s_1tb1tcb = timing_decorator(TCFMM.f3s_1tb1tcb)
  else:
    f3s_1tb1rw = TCFMM.f3s_1tb1rw
    f3s_1tb1rw_scheduled = TCFMM.f3s_1tb1rw_scheduled
    f3s_1tb1tcb = TCFMM.f3s_1tb1tcb

  if args.alg == '1tb1rw':
    final_result, sddmm_result_1tb1rw = f3s_1tb1rw(
      RowWindowOffset, SparseAToXindex, TCblockBitMap, 
      num_nodes, Q, K, V, args.n_warp_per_block, True)
  elif args.alg == '1tb1rw_no_softmax':
    final_result, sddmm_result_1tb1rw = f3s_1tb1rw(
      RowWindowOffset, SparseAToXindex, TCblockBitMap, 
      num_nodes, Q, K, V, args.n_warp_per_block, False)
  elif args.alg == '1tb1rw_scheduled':
    final_result, sddmm_result_1tb1rw_scheduled = f3s_1tb1rw_scheduled(
      RowWindowOffset, sortedRowWindows, SparseAToXindex, TCblockBitMap, 
      num_nodes, Q, K, V, args.n_warp_per_block)
  elif args.alg == '1tb1tcb':
    apply_softmax = True
    save_sddmm_result = False
    final_result, sddmm_result_1tb1tcb = f3s_1tb1tcb(
      RowWindowOffset, SparseAToXindex, TCblockBitMap, 
      num_nodes, Q, K, V, apply_softmax, save_sddmm_result)
  else:
    raise ValueError(f"Invalid algorithm: {args.alg}")
  
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', '-emb', type=int, default=128,
                       help='Dimension of node embeddings')
    parser.add_argument('--n_warp_per_block', '-nw', type=int, default=8,
                       help='Number of warps per block')
    parser.add_argument('--dataset', '-d', type=str, default="reddit",
                       choices=["reddit", "ppa", "protein", "cora", "pubmed"])
    parser.add_argument("--alg", '-a', type=str, default='1tb1rw_scheduled', 
                        choices=['1tb1tcb', '1tb1rw', '1tb1rw_no_softmax', '1tb1rw_scheduled'])
    parser.add_argument("--use_cuda_event", action='store_true', 
                        help='Use CUDA event to measure time, no timer is used otherwise')
    args = parser.parse_args()
    main(args)
