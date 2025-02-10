import argparse
import torch
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from torch_geometric.utils import softmax
import FS_Block
import FS_SDDMM
import FS_SpMM
import dgl.sparse as dglsp

class Perf:
  def __init__(self, algs, datasets):
    self.pd = pd.DataFrame(index=datasets, columns=algs)

datasets = ["reddit", "amazonProducts", "yelp", "amazon0505", "Artist", "Blog", "com-amazon.ungraph", "github.npz", "Ell", "ogbn-products", "citeseer", "pubmed", "cora"]

algs = ['f3s_1tb1tcb', 'f3s_1tb1rw', 'f3s_1tb1rw_no_softmax', 
        'f3s_1tb1rw_scheduled', 'f3s_1tb1rw_scheduled_permuteV',
        'flashSparse_no_softmax', 'flashSparse_naive_softmax', 'flashSparse_stable_softmax', 
        'GTConvFuse_inference_tiling', 'GTConvFuse_inference_hyper']

class GraphInfo:
  def __init__(self, name, row_pointers, column_index, rows, num_nodes, num_edges, disable_dfgnn_hyper=False):
    self.name = name
    self.row_pointers = row_pointers
    self.column_index = column_index
    # row indices of all non-zero elements in the adjacency matrix, required by dfgnn
    self.rows = rows
    assert len(rows) == len(column_index)
    self.num_nodes = num_nodes
    self.num_edges = num_edges
    self.disable_dfgnn_hyper = disable_dfgnn_hyper

# only for flashSparse
class InputInfo:
  def __init__(self):
    self.name = None
    self.row_pointers = None
    self.column_index = None
    self.degrees = None
    self.t_window_rowTensor = None
    self.t_atomicTensor = None
    self.max = None
    self.num_nodes_ori = None
    self.num_nodes = None
    self.num_edges = None
    self.ones = None

def event_timing_decorator(kernel, graphInfo, perf):
  def wrapper(*args, **kwargs):
    niter = 3
    # warmup
    for i in range(3):
      out = kernel(*args, **kwargs)
    torch.cuda.synchronize()
    print(f"{kernel.__name__} warmup done")
    execution_time = 0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for i in range(niter):
      start_event.record()
      out = kernel(*args, **kwargs)
      end_event.record()
      end_event.synchronize()
      execution_time += start_event.elapsed_time(end_event)
    execution_time /= niter
    print(f"{kernel.__name__} average execution time: {execution_time} ms")
    perf.pd.loc[graphInfo.name, kernel.__name__] = execution_time
    return out
  return wrapper

def timing_decorator(kernel, graphInfo, perf):
  def wrapper(*args, **kwargs):
    niter = 3
    # warmup
    for i in range(3):
      kernel(*args, **kwargs)
    total_time = 0
    for i in range(niter):
      output = kernel(*args, **kwargs)
      total_time += output[0].item()
    avg_time = total_time / niter
    print(f"{kernel.__name__} average execution time: {avg_time} ms")
    perf.pd.loc[graphInfo.name, kernel.__name__] = avg_time
    return output
  return wrapper

def flashSparse_no_softmax(X_prime, inputInfo):
  sddmm_time, att = FS_SDDMM.forward_gen_fp16_gnn(   
            X_prime.size(1),                                      
            inputInfo.row_pointers, 
            inputInfo.column_index, 
            inputInfo.degrees, 
            inputInfo.t_window_rowTensor,
            X_prime,X_prime,inputInfo.max)
  spmm_time, h_prime = FS_SpMM.forward_fp16_gnn(   
              inputInfo.row_pointers, 
              inputInfo.column_index, 
              att, 
              inputInfo.t_window_rowTensor,
              inputInfo.t_atomicTensor,
              X_prime, 
              inputInfo.num_nodes, 
              X_prime.size(1), 
              inputInfo.num_nodes_ori)
  total_time = sddmm_time + spmm_time
  return total_time, h_prime

def flashSparse_naive_softmax(X_prime, inputInfo):
  sddmm_time, att = FS_SDDMM.forward_gen_fp16_gnn(   
              X_prime.size(1),                                      
              inputInfo.row_pointers, 
              inputInfo.column_index, 
              inputInfo.degrees, 
              inputInfo.t_window_rowTensor,
              X_prime,X_prime,inputInfo.max)
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)
  start_event.record()
  att = torch.exp(att) # softmax
  end_event.record()
  end_event.synchronize()
  softmax_time = start_event.elapsed_time(end_event)
  spmm_ones_time, rows_sum = FS_SpMM.forward_fp16_gnn_ones(   
              inputInfo.row_pointers, 
              inputInfo.column_index, 
              att, 
              inputInfo.t_window_rowTensor,
              inputInfo.t_atomicTensor,
              inputInfo.ones, 
              inputInfo.num_nodes, 
              inputInfo.ones.size(1), 
              inputInfo.num_nodes_ori)
  spmm_time, h_prime = FS_SpMM.forward_fp16_gnn(   
              inputInfo.row_pointers, 
              inputInfo.column_index, 
              att, 
              inputInfo.t_window_rowTensor,
              inputInfo.t_atomicTensor,
              X_prime, 
              inputInfo.num_nodes, 
              X_prime.size(1), 
              inputInfo.num_nodes_ori)
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)
  start_event.record()
  h_prime = h_prime.div(rows_sum) # softmax
  end_event.record()
  end_event.synchronize()
  softmax_time += start_event.elapsed_time(end_event)
  total_time = sddmm_time + spmm_ones_time + spmm_time + softmax_time
  return total_time, h_prime

def flashSparse_stable_softmax(X_prime, inputInfo, edge_att_rand):
  sddmm_time, att = FS_SDDMM.forward_gen_fp16_gnn(   
              X_prime.size(1),                                      
              inputInfo.row_pointers, 
              inputInfo.column_index, 
              inputInfo.degrees, 
              inputInfo.t_window_rowTensor,
              X_prime,X_prime,inputInfo.max)
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)
  start_event.record()
  softmax(edge_att_rand, ptr=inputInfo.orig_row_pointers.to('cuda'))
  end_event.record()
  end_event.synchronize()
  softmax_time = start_event.elapsed_time(end_event)
  spmm_time, h_prime = FS_SpMM.forward_fp16_gnn(   
              inputInfo.row_pointers, 
              inputInfo.column_index, 
              att, 
              inputInfo.t_window_rowTensor,
              inputInfo.t_atomicTensor,
              X_prime, 
              inputInfo.num_nodes, 
              X_prime.size(1), 
              inputInfo.num_nodes_ori)
  total_time = sddmm_time + softmax_time + spmm_time
  return total_time, h_prime
  
def route_flashSparse(args, inputInfo, Q, perf):
  if args.use_cuda_event:
    no_softmax = timing_decorator(flashSparse_no_softmax, inputInfo, perf)
    naive_softmax = timing_decorator(flashSparse_naive_softmax, inputInfo, perf)
    stable_softmax = timing_decorator(flashSparse_stable_softmax, inputInfo, perf)
  else:
    no_softmax = flashSparse_no_softmax
    naive_softmax = flashSparse_naive_softmax
    stable_softmax = flashSparse_stable_softmax
  if args.alg == "flashSparse_no_softmax" or args.alg == "all":
    no_softmax(Q, inputInfo)
  if args.alg == "flashSparse_naive_softmax" or args.alg == "all":
    naive_softmax(Q, inputInfo)
  if args.alg == "flashSparse_stable_softmax" or args.alg == "all":
    edge_att_rand = torch.rand(inputInfo.num_edges, dtype=torch.float16, device=args.dev)
    print(f"inputInfo.num_edges: {inputInfo.num_edges}, inputInfo.num_nodes: {inputInfo.num_nodes}, inputInfo.max: {inputInfo.max}")
    stable_softmax(Q, inputInfo, edge_att_rand)

def flashSparse_preprocess_dataset(args, graphInfo):
  partSize = 32
  window = 8
  wide = 16
  inputInfo = InputInfo()
  inputInfo.name = graphInfo.name
  inputInfo.orig_row_pointers = graphInfo.row_pointers
  inputInfo.row_pointers, inputInfo.column_index, \
  inputInfo.degrees, inputInfo.t_window_rowTensor, \
  inputInfo.t_atomicTensor = FS_Block.blockProcess_sddmm_balance_gnn(graphInfo.row_pointers.cpu(), graphInfo.column_index.cpu(), window, wide, partSize)
  inputInfo.row_pointers = inputInfo.row_pointers.cuda()
  inputInfo.column_index = inputInfo.column_index.cuda()
  inputInfo.degrees = inputInfo.degrees.cuda()
  inputInfo.t_window_rowTensor = inputInfo.t_window_rowTensor.cuda()
  inputInfo.t_atomicTensor = inputInfo.t_atomicTensor.cuda()
  inputInfo.num_nodes_ori = graphInfo.num_nodes
  if graphInfo.num_nodes%16 !=0 :
    inputInfo.num_nodes = graphInfo.num_nodes + 16 - graphInfo.num_nodes%16
  else:
    inputInfo.num_nodes = graphInfo.num_nodes
  inputInfo.num_edges = graphInfo.num_edges
  inputInfo.ones = torch.ones((inputInfo.num_nodes_ori,1), dtype=torch.float16, device=args.dev)
  max_vectors = torch.max(inputInfo.row_pointers[1:]- inputInfo.row_pointers[:-1])
  if max_vectors%wide > 0 :
      max_vectors += (wide - (max_vectors%wide))
  inputInfo.max = max_vectors / wide
  
  if inputInfo.max % 4 > 0 :
      inputInfo.max += 4 - inputInfo.max%4
  return inputInfo

def route_f3s(args, graphInfo, Q, K, V, perf):
  import TCFMM
  if args.use_cuda_event:
    f3s_1tb1rw = timing_decorator(TCFMM.f3s_1tb1rw, graphInfo, perf)
    f3s_1tb1rw_scheduled= timing_decorator(TCFMM.f3s_1tb1rw_scheduled, graphInfo, perf)
    f3s_1tb1rw_scheduled_permuteV = timing_decorator(TCFMM.f3s_1tb1rw_scheduled_permuteV, graphInfo, perf)
    f3s_1tb1tcb = timing_decorator(TCFMM.f3s_1tb1tcb, graphInfo, perf)
  else:
    f3s_1tb1rw = TCFMM.f3s_1tb1rw
    f3s_1tb1rw_scheduled = TCFMM.f3s_1tb1rw_scheduled
    f3s_1tb1rw_scheduled_permuteV = TCFMM.f3s_1tb1rw_scheduled_permuteV
    f3s_1tb1tcb = TCFMM.f3s_1tb1tcb

  if args.alg == 'f3s_1tb1tcb' or args.alg == 'all':
    RowWindowOffset, sortedRowWindows, TCblockRowid,\
    TCblocktileId, TCblockoffset, SparseAToXindex,\
    TBBoundaries, TCblockBitMap, block_count = f3s_preprocess_dataset(args, graphInfo, BLK_W=8)
    print("f3s_1tb1tcb")
    apply_softmax = True
    save_sddmm_result = False
    f3s_1tb1tcb(
      RowWindowOffset, SparseAToXindex, TCblockBitMap, 
      graphInfo.num_nodes, Q, K, V, apply_softmax, save_sddmm_result)
    
  RowWindowOffset, sortedRowWindows, TCblockRowid,\
  TCblocktileId, TCblockoffset, SparseAToXindex,\
  TBBoundaries, TCblockBitMap, block_count = f3s_preprocess_dataset(args, graphInfo, BLK_W=16)
  if args.alg == 'f3s_1tb1rw' or args.alg == 'all':
    print("f3s_1tb1rw")
    f3s_1tb1rw(
      RowWindowOffset, SparseAToXindex, TCblockBitMap, 
      graphInfo.num_nodes, Q, K, V, args.n_warp_per_block, True)
  if args.alg == 'f3s_1tb1rw_scheduled' or args.alg == 'all':
    print("f3s_1tb1rw_scheduled")
    f3s_1tb1rw_scheduled(
      RowWindowOffset, sortedRowWindows, SparseAToXindex, TCblockBitMap, 
      graphInfo.num_nodes, Q, K, V, args.n_warp_per_block)
  if args.alg == 'f3s_1tb1rw_scheduled_permuteV' or args.alg == 'all':
    print("f3s_1tb1rw_scheduled_permuteV")
    f3s_1tb1rw_scheduled_permuteV(
      RowWindowOffset, sortedRowWindows, SparseAToXindex, TCblockBitMap, 
      graphInfo.num_nodes, Q, K, V, args.n_warp_per_block)

def f3s_preprocess_dataset(args, graphInfo, BLK_W):
  from TCFMM import preprocess_gpu
  BLK_H = 16
  # Set up tensors for preprocessing
  num_row_windows = (graphInfo.num_nodes + BLK_H - 1) // BLK_H
  # Move tensors to GPU
  blockPartition_cuda = torch.zeros(num_row_windows, dtype=torch.int).cuda()
  edgeToColumn_cuda = torch.zeros(graphInfo.num_edges, dtype=torch.int).cuda()
  edgeToRow_cuda = torch.zeros(graphInfo.num_edges, dtype=torch.int).cuda()
  return preprocess_gpu(graphInfo.column_index, graphInfo.row_pointers, graphInfo.num_nodes, 
                        BLK_H, BLK_W, blockPartition_cuda, 
                        edgeToColumn_cuda, edgeToRow_cuda)

def route_dfgnn(args, graphInfo, Q, K, V, perf):
  from DFGNN.operators.fused_gtconv import GTConvFuse_inference_tiling, GTConvFuse_inference_hyper
  if args.use_cuda_event:
    dfgnn_tiling = event_timing_decorator(GTConvFuse_inference_tiling, graphInfo, perf)
    dfgnn_hyper = event_timing_decorator(GTConvFuse_inference_hyper, graphInfo, perf)
  else:
    dfgnn_tiling = GTConvFuse_inference_tiling
    dfgnn_hyper = GTConvFuse_inference_hyper

  if args.alg == "dfgnn_tiling" or args.alg == "all":
    smem_consume, val, Q, K, V = dfgnn_preprocess_dataset(args, graphInfo, Q, K, V, alg="dfgnn_tiling")
    out = dfgnn_tiling(graphInfo.row_pointers, graphInfo.column_index, val, smem_consume, Q, K, V)
  if args.alg == "dfgnn_hyper" or args.alg == "all" and not graphInfo.disable_dfgnn_hyper:
    smem_consume, val, Q, K, V = dfgnn_preprocess_dataset(args, graphInfo, Q, K, V, alg="dfgnn_hyper")
    out = dfgnn_hyper(graphInfo.row_pointers, graphInfo.column_index, graphInfo.rows, val, smem_consume, Q, K, V)

def dfgnn_preprocess_dataset(args, graphInfo, Q, K, V, alg):
  # add 1 dimension of 1 to Q, K, V
  Q = Q.unsqueeze(1).float()
  K = K.unsqueeze(1).float()
  V = V.unsqueeze(1).float()
  max_neigh = 128 # according to DF-GNN/DFGNN/layers/util.py
  WARP_SIZE = 32
  val = torch.ones(graphInfo.num_edges, dtype=torch.float32, device=args.dev)
  if alg == "dfgnn_tiling":
    smem_consume = (max_neigh + WARP_SIZE - 1) // WARP_SIZE * WARP_SIZE
    return smem_consume, val, Q, K, V
  elif alg == "dfgnn_hyper":
    smem_consume = (max_neigh * 8 + WARP_SIZE - 1) // WARP_SIZE * WARP_SIZE
    return smem_consume, val, Q, K, V
  else:
    raise ValueError(f"Invalid algorithm: {alg}")
  
def route_methods(args, graphInfo, Q, K, V, perf):
  # load dataset
  if args.method == "f3s" or args.method == "all":
    route_f3s(args, graphInfo, Q, K, V, perf)
  if args.method == "flashSparse" or args.method == "all":
    # num_features and num_classes are for creating inputInfo.x and .y, which are not used in the kernel
    inputInfo = flashSparse_preprocess_dataset(args, graphInfo)
    route_flashSparse(args, inputInfo, Q, perf)
  if args.method == "df-gnn" or args.method == "all":
    route_dfgnn(args, graphInfo, Q, K, V, perf)

def load_dataset(dataset_name):
  print(f"===========loading dataset: {dataset_name}===========")
  path = f"/share/crsp/lab/amowli/share/Fused3S/dataset/{dataset_name}.npz"
  dataset = np.load(path)
  src_li = dataset['src_li'] # this can contain duplicate edges
  dst_li = dataset['dst_li']
  val = [1] * len(src_li)
  edge_index = np.stack([src_li, dst_li])
  scipy_coo = coo_matrix((val, edge_index), shape=(dataset['num_nodes'], dataset['num_nodes']))
  adj = scipy_coo.tocsr()
  row_pointers = torch.IntTensor(adj.indptr).cuda()
  column_index = torch.IntTensor(adj.indices).cuda()
  num_nodes = dataset['num_nodes']
  num_edges = column_index.shape[0]
  # row indices of all non-zero elements in the adjacency matrix, required by dfgn
  # A = dglsp.spmatrix(torch.tensor(edge_index), shape=(num_nodes, num_nodes))
  # rows = A.row.int().cuda()
  # rows = torch.sort(rows).values
  row_nnz = np.diff(adj.indptr)
  disable_dfgnn_hyper = False
  if np.max(row_nnz) > 128:
    print(f"max row_nnz: {np.max(row_nnz)} greater than 128, dfgnn_hyper is disabled")
    disable_dfgnn_hyper = True
  row_indices = np.repeat(np.arange(adj.shape[0]), row_nnz)
  rows = torch.IntTensor(row_indices).cuda()
  # assert torch.all(rows == rows_alt), "rows_alt is different from rows"
  graphInfo = GraphInfo(dataset_name, row_pointers, column_index, rows, num_nodes, num_edges, disable_dfgnn_hyper)
  return graphInfo

def main(args):
  num_heads = 1
  args.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  if args.dataset != "all":
    test_dataset = [args.dataset]
  else:
    test_dataset = datasets
  if args.method == "all":
    test_algs = algs
  else:
    test_algs = [args.method]
  perf = Perf(test_algs, test_dataset)
  for dataset in test_dataset:
    graphInfo = load_dataset(dataset)
    print(f"dataset: {dataset}, num_nodes: {graphInfo.num_nodes}, num_edges: {graphInfo.num_edges}")
    Q = torch.rand(graphInfo.num_nodes, args.embedding_dim, dtype=torch.float16, device=args.dev)
    K = torch.rand(graphInfo.num_nodes, args.embedding_dim, dtype=torch.float16, device=args.dev)
    V = torch.rand(graphInfo.num_nodes, args.embedding_dim, dtype=torch.float16, device=args.dev)
    route_methods(args, graphInfo, Q, K, V, perf)
  print(perf.pd)
  perf.pd.to_csv(f"baseline_comp_kernel_only_{args.method}_{args.alg}.csv")
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', '-emb', type=int, default=128,
                       help='Dimension of node embeddings')
    parser.add_argument('--n_warp_per_block', '-nw', type=int, default=8,
                       help='Number of warps per block')
    parser.add_argument('--dataset', '-d', type=str, default="reddit",
                       choices=datasets + ["all"])
    parser.add_argument('--method', '-m', type=str, default="f3s",
                       choices=["f3s", "flashSparse", "df-gnn", "all"])
    parser.add_argument("--alg", '-a', type=str, default='f3s_1tb1rw_scheduled', 
                        choices= algs + ['all'])
    parser.add_argument("--use_cuda_event", action='store_true', 
                        help='Use CUDA event to measure time, no timer is used otherwise')
    args = parser.parse_args()
    main(args)
