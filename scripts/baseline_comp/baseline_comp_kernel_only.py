import argparse
import torch
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from torch_geometric.utils import softmax
import FS_Block
import FS_SDDMM
import FS_SpMM
import ogb

def check_gpu_memory():
  if torch.cuda.is_available():
    total_mem = torch.cuda.get_device_properties(0).total_memory
    reserved_mem = torch.cuda.memory_reserved(0)
    allocated_mem = torch.cuda.memory_allocated(0)
    free_mem = total_mem - reserved_mem
    print(f"GPU memory: total={total_mem/1e9:.2f}GB, reserved={reserved_mem/1e9:.2f}GB, allocated={allocated_mem/1e9:.2f}GB, free={free_mem/1e9:.2f}GB")

class Perf:
  def __init__(self, algs, datasets):
    self.pd = pd.DataFrame(index=datasets, columns=algs)

datasets = ["reddit", "amazonProducts", "yelp", "amazon0505", 
            "Artist", "Blog", "com-amazon.ungraph", "github", 
            "Ell", "ogbn-products", "citeseer", "pubmed", "cora",
            "igb_small", "igb_medium"]

algs = ['f3s_1tb1tcb', 'f3s_1tb1rw', 'f3s_1tb1rw_no_softmax', 
        'f3s_1tb1rw_scheduled', 'f3s_1tb1rw_scheduled_permuteV',
        'flashSparse_no_softmax', 'flashSparse_naive_softmax', 'flashSparse_stable_softmax', 
        'GTConvFuse_inference_tiling', 'GTConvFuse_inference_hyper']

class GraphInfo:
  # adj is a scipy.sparse.csr_matrix
  def __init__(self, name, adj):
    self.name = name
    self.adj = adj
    self.num_nodes = adj.shape[0]
    self.num_edges = adj.nnz

  def get_row_pointers(self):
    return torch.IntTensor(self.adj.indptr).cuda()

  def get_column_index(self):
    return torch.IntTensor(self.adj.indices).cuda()

  def get_edge_index(self):
    coo = self.adj.tocoo()
    rowIndex = torch.LongTensor(coo.row).cuda()
    colIndex = torch.LongTensor(coo.col).cuda()
    return torch.stack((rowIndex, colIndex))

  def get_rows(self):
    # row indices of all non-zero elements in the adjacency matrix, required by dfgnn
    row_nnz = np.diff(self.adj.indptr)
    row_indices = np.repeat(np.arange(self.num_nodes), row_nnz)
    rows = torch.IntTensor(row_indices).cuda()
    return rows

  def enable_dfgnn_hyper(self):
    row_nnz = np.diff(self.adj.indptr)
    if np.max(row_nnz) > 128: # 128 is hardcoded in DFGNN/DFGNN/layers/util.py
      print(f"max row_nnz: {np.max(row_nnz)} greater than 128, dfgnn_hyper is disabled")
      return False
    return True

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
    times = []
    niter = 10
    # warmup
    for i in range(3):
      out = kernel(*args, **kwargs)
    torch.cuda.synchronize()
    print(f"{kernel.__name__} warmup done")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for i in range(niter):
      start_event.record()
      out = kernel(*args, **kwargs)
      end_event.record()
      end_event.synchronize()
      times.append(start_event.elapsed_time(end_event))
    execution_time = np.median(times)
    print(f"{kernel.__name__} median execution time: {execution_time} ms")
    perf.pd.loc[graphInfo.name, kernel.__name__] = execution_time
    return out
  return wrapper

def timing_decorator(kernel, graphInfo, perf):
  def wrapper(*args, **kwargs):
    times = []
    niter = 10
    # warmup
    for i in range(3):
      kernel(*args, **kwargs)
    for i in range(niter):
      output = kernel(*args, **kwargs)
      times.append(output[0].item())
    avg_time = np.median(times)
    print(f"{kernel.__name__} median execution time: {avg_time} ms")
    perf.pd.loc[graphInfo.name, kernel.__name__] = avg_time
    return output
  return wrapper

def flashSparse_no_softmax(Q, K, V, inputInfo):
  sddmm_time, att = FS_SDDMM.forward_gen_fp16_gnn(   
                      Q.size(1),                                      
                      inputInfo.row_pointers, 
                      inputInfo.column_index, 
                      inputInfo.degrees, 
                      inputInfo.t_window_rowTensor,
                      Q,K,inputInfo.max)
  spmm_time, h_prime = FS_SpMM.forward_fp16_gnn(   
                        inputInfo.row_pointers, 
                        inputInfo.column_index, 
                        att, 
                        inputInfo.t_window_rowTensor,
                        inputInfo.t_atomicTensor,
                        V, 
                        inputInfo.num_nodes, 
                        V.size(1), 
                        inputInfo.num_nodes_ori)
  total_time = sddmm_time + spmm_time
  return total_time, h_prime

def flashSparse_naive_softmax(Q, K, V, inputInfo):
  sddmm_time, att = FS_SDDMM.forward_gen_fp16_gnn(   
                      Q.size(1),                                      
                      inputInfo.row_pointers, 
                      inputInfo.column_index, 
                      inputInfo.degrees, 
                      inputInfo.t_window_rowTensor,
                      Q,K,inputInfo.max)
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)
  start_event.record()
  att = torch.exp(att) # softmax
  end_event.record()
  end_event.synchronize()
  exp_time = start_event.elapsed_time(end_event)
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
                        V, 
                        inputInfo.num_nodes, 
                        V.size(1), 
                        inputInfo.num_nodes_ori)
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)
  start_event.record()
  h_prime = h_prime.div(rows_sum) # softmax
  end_event.record()
  end_event.synchronize()
  div_time = start_event.elapsed_time(end_event)
  total_time = sddmm_time + spmm_ones_time + spmm_time + exp_time + div_time
  print(f"naive_softmax: sddmm: {sddmm_time} ms, spmm_ones: {spmm_ones_time} ms, spmm: {spmm_time} ms, exp: {exp_time} ms, div: {div_time} ms, total: {total_time} ms")
  return total_time, h_prime

def flashSparse_stable_softmax(Q, K, V, inputInfo):
  sddmm_time, att = FS_SDDMM.forward_gen_fp16_gnn(   
              Q.size(1),                                      
              inputInfo.row_pointers, 
              inputInfo.column_index, 
              inputInfo.degrees, 
              inputInfo.t_window_rowTensor,
              Q,K,inputInfo.max)
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)
  start_event.record()
  softmax(att[:inputInfo.num_edges], ptr=inputInfo.orig_row_pointers, dim=0)
  end_event.record()
  end_event.synchronize()
  softmax_time = start_event.elapsed_time(end_event)
  spmm_time, h_prime = FS_SpMM.forward_fp16_gnn(   
              inputInfo.row_pointers, 
              inputInfo.column_index, 
              att, 
              inputInfo.t_window_rowTensor,
              inputInfo.t_atomicTensor,
              V, 
              inputInfo.num_nodes, 
              V.size(1), 
              inputInfo.num_nodes_ori)
  total_time = sddmm_time + softmax_time + spmm_time
  print(f"stable_softmax: sddmm: {sddmm_time} ms, softmax: {softmax_time} ms, spmm: {spmm_time} ms, total: {total_time} ms")
  return total_time, h_prime
  
def route_flashSparse(args, inputInfo, perf):
  Q = torch.rand(inputInfo.num_nodes, args.embedding_dim, dtype=torch.float16, device=args.dev)
  K = torch.rand(inputInfo.num_nodes, args.embedding_dim, dtype=torch.float16, device=args.dev)
  V = torch.rand(inputInfo.num_nodes, args.embedding_dim, dtype=torch.float16, device=args.dev)
  if args.use_cuda_event:
    no_softmax = timing_decorator(flashSparse_no_softmax, inputInfo, perf)
    naive_softmax = timing_decorator(flashSparse_naive_softmax, inputInfo, perf)
    stable_softmax = timing_decorator(flashSparse_stable_softmax, inputInfo, perf)
  else:
    no_softmax = flashSparse_no_softmax
    naive_softmax = flashSparse_naive_softmax
    stable_softmax = flashSparse_stable_softmax

  if args.alg == "flashSparse_no_softmax":
    try:
      no_softmax(Q, K, V, inputInfo)
    except Exception as e:
      print(f"Error in flashSparse_no_softmax: {e}")
  if args.alg == "flashSparse_naive_softmax" or args.alg == "all":
    try:
      naive_softmax(Q, K, V, inputInfo)
    except Exception as e:
      print(f"Error in flashSparse_naive_softmax: {e}")
  if args.alg == "flashSparse_stable_softmax" or args.alg == "all":
    try:
      stable_softmax(Q, K, V, inputInfo)
    except Exception as e:
      print(f"Error in flashSparse_stable_softmax: {e}")

def flashSparse_preprocess_dataset(args, graphInfo):
  partSize = 32
  window = 8
  wide = 16
  inputInfo = InputInfo()
  inputInfo.name = graphInfo.name
  inputInfo.orig_row_pointers = graphInfo.get_row_pointers()
  inputInfo.column_index = graphInfo.get_column_index()
  inputInfo.row_pointers, inputInfo.column_index, \
  inputInfo.degrees, inputInfo.t_window_rowTensor, \
  inputInfo.t_atomicTensor = FS_Block.blockProcess_sddmm_balance_gnn(inputInfo.orig_row_pointers.cpu(),
                                                                     inputInfo.column_index.cpu(), 
                                                                     window, wide, partSize)
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

def route_f3s(args, graphInfo, perf):
  Q = torch.rand(graphInfo.num_nodes, args.embedding_dim, dtype=torch.float16, device=args.dev)
  K = torch.rand(graphInfo.num_nodes, args.embedding_dim, dtype=torch.float16, device=args.dev)
  V = torch.rand(graphInfo.num_nodes, args.embedding_dim, dtype=torch.float16, device=args.dev)
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
    apply_softmax = True
    try:
      f3s_1tb1rw(
        RowWindowOffset, SparseAToXindex, TCblockBitMap, 
        graphInfo.num_nodes, Q, K, V, args.n_warp_per_block, apply_softmax,
        args.check_sm_active_time)
    except Exception as e:
      print(f"Error in f3s_1tb1rw: {e}")
  if args.alg == 'f3s_1tb1rw_scheduled' or args.alg == 'all':
    print("f3s_1tb1rw_scheduled")
    try:
      f3s_1tb1rw_scheduled(
        RowWindowOffset, sortedRowWindows, SparseAToXindex, TCblockBitMap, 
        graphInfo.num_nodes, Q, K, V, args.n_warp_per_block, 
        args.check_sm_active_time)
    except Exception as e:
      print(f"Error in f3s_1tb1rw_scheduled: {e}")
  if args.alg == 'f3s_1tb1rw_scheduled_permuteV' or args.alg == 'all':
    print("f3s_1tb1rw_scheduled_permuteV")
    try:
      f3s_1tb1rw_scheduled_permuteV(
        RowWindowOffset, sortedRowWindows, SparseAToXindex, TCblockBitMap, 
        graphInfo.num_nodes, Q, K, V, args.n_warp_per_block)
    except Exception as e:
      print(f"Error in f3s_1tb1rw_scheduled_permuteV: {e}")

def f3s_preprocess_dataset(args, graphInfo, BLK_W):
  from TCFMM import preprocess_gpu
  BLK_H = 16
  # Set up tensors for preprocessing
  num_row_windows = (graphInfo.num_nodes + BLK_H - 1) // BLK_H
  # Move tensors to GPU
  blockPartition_cuda = torch.zeros(num_row_windows, dtype=torch.int).cuda()
  edgeToColumn_cuda = torch.zeros(graphInfo.num_edges, dtype=torch.int).cuda()
  edgeToRow_cuda = torch.zeros(graphInfo.num_edges, dtype=torch.int).cuda()
  column_index = graphInfo.get_column_index()
  row_pointers = graphInfo.get_row_pointers()
  return preprocess_gpu(column_index, row_pointers, graphInfo.num_nodes, 
                        BLK_H, BLK_W, blockPartition_cuda, 
                        edgeToColumn_cuda, edgeToRow_cuda)

def route_dfgnn(args, graphInfo, perf):
  from DFGNN.operators.fused_gtconv import GTConvFuse_inference_tiling, GTConvFuse_inference_hyper
  if args.use_cuda_event:
    dfgnn_tiling = event_timing_decorator(GTConvFuse_inference_tiling, graphInfo, perf)
    dfgnn_hyper = event_timing_decorator(GTConvFuse_inference_hyper, graphInfo, perf)
  else:
    dfgnn_tiling = GTConvFuse_inference_tiling
    dfgnn_hyper = GTConvFuse_inference_hyper

  num_heads = 1
  Q = torch.rand(graphInfo.num_nodes, num_heads, args.embedding_dim, dtype=torch.float32, device=args.dev)
  K = torch.rand(graphInfo.num_nodes, num_heads, args.embedding_dim, dtype=torch.float32, device=args.dev)
  V = torch.rand(graphInfo.num_nodes, num_heads, args.embedding_dim, dtype=torch.float32, device=args.dev)
  row_pointers = graphInfo.get_row_pointers()
  column_index = graphInfo.get_column_index()
  if args.alg == "dfgnn_tiling" or args.alg == "all":
    smem_consume, val = dfgnn_preprocess_dataset(args, graphInfo, alg="dfgnn_tiling")
    try:
      out = dfgnn_tiling(row_pointers, column_index, val, smem_consume, Q, K, V)
    except Exception as e:
      print(f"Error in dfgnn_tiling: {e}")
  if (args.alg == "dfgnn_hyper" or args.alg == "all") and graphInfo.enable_dfgnn_hyper():
    smem_consume, val = dfgnn_preprocess_dataset(args, graphInfo, alg="dfgnn_hyper")
    rows = graphInfo.get_rows()
    try:
      out = dfgnn_hyper(row_pointers, column_index, rows, val, smem_consume, Q, K, V)
    except Exception as e:
      print(f"Error in dfgnn_hyper: {e}")

def dfgnn_preprocess_dataset(args, graphInfo, alg):
  max_neigh = 128 # according to DF-GNN/DFGNN/layers/util.py
  WARP_SIZE = 32
  val = torch.ones(graphInfo.num_edges, dtype=torch.float32, device=args.dev)
  if alg == "dfgnn_tiling":
    smem_consume = (max_neigh + WARP_SIZE - 1) // WARP_SIZE * WARP_SIZE
    return smem_consume, val
  elif alg == "dfgnn_hyper":
    smem_consume = (max_neigh * 8 + WARP_SIZE - 1) // WARP_SIZE * WARP_SIZE
    return smem_consume, val
  else:
    raise ValueError(f"Invalid algorithm: {alg}")
  
def pyg_gtconv(args, graphInfo, perf):
  from torch_geometric.nn import TransformerConv
  conv = TransformerConv(args.embedding_dim, args.embedding_dim, heads=1, bias=False, root_weight=False)
  propagate = event_timing_decorator(conv.propagate, graphInfo, perf)
  num_nodes = graphInfo.num_nodes
  Q = torch.rand(num_nodes, 1, args.embedding_dim, dtype=torch.float32, device=args.dev)
  K = torch.rand(num_nodes, 1, args.embedding_dim, dtype=torch.float32, device=args.dev)
  V = torch.rand(num_nodes, 1, args.embedding_dim, dtype=torch.float32, device=args.dev)
  edge_index = graphInfo.get_edge_index().contiguous()
  try:
    propagate(edge_index=edge_index, query=Q, key=K, value=V, edge_attr=None)
  except Exception as e:
    print(f"Error in pyg_gtconv: {e}")

def route_methods(args, graphInfo, perf):
  if args.method == "flashSparse" or args.method == "all":
    # num_features and num_classes are for creating inputInfo.x and .y, which are not used in the kernel
    inputInfo = flashSparse_preprocess_dataset(args, graphInfo)
    torch.cuda.empty_cache()
    check_gpu_memory()
    route_flashSparse(args, inputInfo, perf)
    del inputInfo
  if args.method == "f3s" or args.method == "all":
    torch.cuda.empty_cache()
    check_gpu_memory()
    route_f3s(args, graphInfo, perf)
  if args.method == "df-gnn" or args.method == "all":
    torch.cuda.empty_cache()
    check_gpu_memory()
    route_dfgnn(args, graphInfo, perf)
  if args.method == "pyg" or args.method == "all":
    torch.cuda.empty_cache()
    check_gpu_memory()
    pyg_gtconv(args, graphInfo, perf)

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
  graphInfo = GraphInfo(dataset_name, adj)
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
    check_gpu_memory()
    route_methods(args, graphInfo, perf)
  print(perf.pd)
  perf.pd.to_csv(f"baseline_comp_kernel_only_{args.method}_{args.alg}_{args.dataset}.csv")
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', '-emb', type=int, default=128,
                       help='Dimension of node embeddings')
    parser.add_argument('--n_warp_per_block', '-nw', type=int, default=8,
                       help='Number of warps per block')
    parser.add_argument('--dataset', '-d', type=str, default="reddit",
                       choices=datasets + ["all"])
    parser.add_argument('--method', '-m', type=str, default="f3s",
                       choices=["f3s", "flashSparse", "df-gnn", "pyg", "all"])
    parser.add_argument("--alg", '-a', type=str, default='f3s_1tb1rw_scheduled', 
                        choices= algs + ['all'])
    parser.add_argument("--use_cuda_event", action='store_true', 
                        help='Use CUDA event to measure time, runs multiple iterations to get average time')
    parser.add_argument("--check_sm_active_time", action='store_true', 
                        help='Check SM active time, only valid for f3s_1tb1rw_scheduled and f3s_1tb1rw')
    args = parser.parse_args()
    main(args)
