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
from typing import Optional, Union, Tuple, overload
import math
import torch_geometric
import torch.nn.functional as F
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptTensor,
    PairTensor,
    SparseTensor,
)

# datasets = ["reddit", "amazonProducts", "yelp", "amazon0505", 
#             "Artist", "Blog", "com-amazon.ungraph", "github", 
#             "Ell", "ogbn-products", "citeseer", "pubmed", "cora",
#             "igb_small", "igb_medium"]

datasets = ["citeseer", "cora", "pubmed", "Ell", "github", 
            "Artist", "com-amazon.ungraph", "Blog", 
            "amazon0505", "igb_small", "yelp", "reddit", 
            "igb_medium", "ogbn-products", "amazonProducts"]

algs = ['f3s_1tb1tcb', 'f3s_1tb1rw', 'f3s_1tb1rw_no_softmax', 
        'f3s_1tb1rw_scheduled', 'f3s_1tb1rw_scheduled_permuteV',
        'flashSparse_no_softmax', 'flashSparse_naive_softmax', 'flashSparse_stable_softmax', 
        'dfgnn_tiling', 'dfgnn_hyper', 'pyg_gtconv']

def check_gpu_memory():
  if torch.cuda.is_available():
    total_mem = torch.cuda.get_device_properties(0).total_memory
    reserved_mem = torch.cuda.memory_reserved(0)
    allocated_mem = torch.cuda.memory_allocated(0)
    free_mem = total_mem - reserved_mem
    print(f"GPU memory: total={total_mem/1e9:.2f}GB, reserved={reserved_mem/1e9:.2f}GB, allocated={allocated_mem/1e9:.2f}GB, free={free_mem/1e9:.2f}GB")

def get_gpu_model():
  if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    # Extract just the model name (A100, H100, etc.)
    if 'A100' in gpu_name:
      return 'A100'
    elif 'H100' in gpu_name:
      return 'H100'
    elif 'A30' in gpu_name:
      return 'A30'
    elif 'V100' in gpu_name:
      return 'V100'
    elif 'T4' in gpu_name:
      return 'T4'
    else:
      # Return the full name if we don't recognize the model
      return gpu_name.replace(' ', '_')
  return 'unknown'

class Perf:
  def __init__(self, algs, datasets):
    self.runtime_pd = pd.DataFrame(index=datasets, columns=algs)
    self.throughput_pd = pd.DataFrame(index=datasets, columns=algs)
    self.bandwidth_pd = pd.DataFrame(index=datasets, columns=algs)
    # Set "dataset" as the name for the index
    self.runtime_pd.index.name = "dataset"
    self.throughput_pd.index.name = "dataset"
    self.bandwidth_pd.index.name = "dataset"

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

  def enable_dfgnn_hyper(self, gpu='A30'):
    self.max_row_nnz = np.max(np.diff(self.adj.indptr))
    # shm_size in bytes
    if gpu == 'A30':
      shm_size = 160000/4
    if gpu == 'H100':
      shm_size = 228000/4
    # each warp takes 1 row. 8 warps per block.
    if self.max_row_nnz * 8 * 4 > shm_size:
      print(f"shm required: {self.max_row_nnz * 8 * 4} bytes is greater than {shm_size}B available on {gpu}, dfgnn_hyper is disabled")
      return False
    return True
    # row_nnz = np.diff(self.adj.indptr)
    # if np.max(row_nnz) > 128: # 128 is hardcoded in DFGNN/DFGNN/layers/util.py
    #   print(f"max row_nnz: {np.max(row_nnz)} greater than 128, dfgnn_hyper is disabled")
    #   return False
    # return True
  
  def dfgnn_flop_and_lgd(self, embedding_dim):
    sddmm_flops = self.adj.nnz * embedding_dim * 2
    # max, subtract, exp, sum, div. each one is self.adj.nnz flops
    softmax_flops = self.adj.nnz * 5
    spmm_flops = self.adj.nnz * embedding_dim * 2
    self.num_flops_dfgnn = spmm_flops + sddmm_flops + softmax_flops
    # 2 embedding_dim vector needed to compute each nnz in S, in fp32
    S = self.adj.nnz * embedding_dim * 2 * 4 
    # 1 embedding_dim vector needed to compute each element in O, in fp32
    O = self.adj.nnz * embedding_dim * 4
    self.lgd_dfgnn = S + O

  def f3s_flop_and_lgd(self, rowWindowOffset, embedding_dim, use_1tb1tcb):
    n_tcb = torch.sum(torch.diff(rowWindowOffset)).item()
    BLK_H = 16
    if use_1tb1tcb:
      BLK_W = 8
    else:
      BLK_W = 16
    nnz = n_tcb * BLK_H * BLK_W
    sddmm = nnz * embedding_dim * 2
    softmax = nnz * 5
    spmm = nnz * embedding_dim * 2
    # f3s global memory access in bytes
    Q = self.num_nodes * embedding_dim * 2
    K = n_tcb * BLK_W * embedding_dim * 2
    V = n_tcb * BLK_W * embedding_dim * 2
    O = n_tcb * BLK_W * embedding_dim * 4
    if use_1tb1tcb:
      self.num_flops_f3s_1tb1tcb = sddmm + spmm + softmax
      self.lgd_f3s_1tb1tcb = Q + K + V + O
    else:
      self.num_flops_f3s_1tb1rw = sddmm + spmm + softmax
      self.lgd_f3s_1tb1rw = Q + K + V + O

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

  def flashSparse_flop_and_lgd(self, row_pointers, embedding_dim):
    BLK_H = 16
    BLK_W = 8
    nvectors_each_row_window = torch.diff(row_pointers)
    # take the ceiling of each element divided by 16
    nvectors_each_row_window = torch.ceil(nvectors_each_row_window / 16)
    # sum all elements in nvectors_each_row_window
    n_tcb = torch.sum(nvectors_each_row_window).item()
    nnz = n_tcb * BLK_H * BLK_W
    sddmm = nnz * embedding_dim * 2
    # exp, div, sum, no max no subtract
    softmax = nnz * 3
    spmm = nnz * embedding_dim * 2
    self.num_flops_flashSparse = sddmm + spmm + softmax
    Q = self.num_nodes * embedding_dim * 2
    K = n_tcb * BLK_H * embedding_dim * 2
    # *2 because S is wrote into HBM and read back
    S = n_tcb * BLK_H * BLK_W * 4 * 2 
    V = n_tcb * BLK_H * embedding_dim * 2
    O = self.num_nodes * embedding_dim * 4
    self.lgd_flashSparse = Q + K + S + V + O

def event_timing_decorator(kernel, kernel_name, graphInfo, perf):
  def wrapper(*args, **kwargs):
    times = []
    throughput = []
    bandwidth = []
    niter = 10
    # warmup
    for i in range(3):
      out = kernel(*args, **kwargs)
    torch.cuda.synchronize()
    print(f"{kernel_name} warmup done")
    flop = graphInfo.num_flops_dfgnn
    lgd = graphInfo.lgd_dfgnn
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for i in range(niter):
      start_event.record()
      kernel(*args, **kwargs)
      end_event.record()
      end_event.synchronize()
      times.append(start_event.elapsed_time(end_event))
      # x1000 to convert to FLOPS and B/s since time is in ms
      throughput.append(flop*1000 / times[-1])
      bandwidth.append(lgd*1000 / times[-1])
    median_time = np.median(times)
    avg_time = np.average(times)
    std_time = np.std(times)
    median_throughput = np.median(throughput)
    avg_throughput = np.average(throughput)
    std_throughput = np.std(throughput)
    median_bandwidth = np.median(bandwidth)
    avg_bandwidth = np.average(bandwidth)
    std_bandwidth = np.std(bandwidth)
    print(f"{kernel_name} execution time (ms). median: {median_time}, average: {avg_time}, standard deviation: {std_time}")
    print(f"{kernel_name} throughput (FLOPS). median: {median_throughput}, average: {avg_throughput}, standard deviation: {std_throughput}")
    print(f"{kernel_name} bandwidth (B/s). median: {median_bandwidth}, average: {avg_bandwidth}, standard deviation: {std_bandwidth}")
    perf.runtime_pd.loc[graphInfo.name, kernel_name] = median_time
    perf.throughput_pd.loc[graphInfo.name, kernel_name] = median_throughput
    perf.bandwidth_pd.loc[graphInfo.name, kernel_name] = median_bandwidth
    return out
  return wrapper

def timing_decorator(kernel, kernel_name, graphInfo, perf):
  def wrapper(*args, **kwargs):
    times = []
    throughput = []
    bandwidth = []
    niter = 10
    flop = 0
    lgd = 0
    if kernel_name == "flashSparse_stable_softmax" \
    or kernel_name == "flashSparse_naive_softmax" \
    or kernel_name == "flashSparse_no_softmax":
      flop = graphInfo.num_flops_flashSparse
      lgd = graphInfo.lgd_flashSparse
    elif kernel_name == "f3s_1tb1rw" \
    or kernel_name == "f3s_1tb1rw_scheduled" \
    or kernel_name == "f3s_1tb1rw_scheduled_permuteV":
      flop = graphInfo.num_flops_f3s_1tb1rw
      lgd = graphInfo.lgd_f3s_1tb1rw
    elif kernel_name == "f3s_1tb1tcb":
      flop = graphInfo.num_flops_f3s_1tb1tcb
      lgd = graphInfo.lgd_f3s_1tb1tcb
    else:
      raise ValueError(f"Unknown kernel name: {kernel_name}")
    # warmup
    for i in range(3):
      kernel(*args, **kwargs)
    for i in range(niter):
      output = kernel(*args, **kwargs)
      times.append(output[0].item())
      # x1000 to convert to FLOPS and B/s since time is in ms
      throughput.append(flop*1000 / times[-1])
      bandwidth.append(lgd*1000 / times[-1])
    median_time = np.median(times)
    avg_time = np.average(times)
    std_time = np.std(times)
    median_throughput = np.median(throughput)
    avg_throughput = np.average(throughput)
    std_throughput = np.std(throughput)
    median_bandwidth = np.median(bandwidth)
    avg_bandwidth = np.average(bandwidth)
    std_bandwidth = np.std(bandwidth)
    print(f"{kernel_name} median execution time: {median_time} ms, avg execution time: {avg_time} ms, std execution time: {std_time} ms")
    print(f"{kernel_name} median throughput (FLOPS): {median_throughput}, avg throughput: {avg_throughput}, std throughput: {std_throughput}")
    print(f"{kernel_name} median bandwidth (B/s): {median_bandwidth}, avg bandwidth: {avg_bandwidth}, std bandwidth: {std_bandwidth}")
    perf.runtime_pd.loc[graphInfo.name, kernel_name] = median_time
    perf.throughput_pd.loc[graphInfo.name, kernel_name] = median_throughput
    perf.bandwidth_pd.loc[graphInfo.name, kernel_name] = median_bandwidth
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
  row_pointers = inputInfo.orig_row_pointers.long()
  start_event.record()
  softmax(att[:inputInfo.num_edges], ptr=row_pointers, dim=0)
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
    no_softmax = timing_decorator(flashSparse_no_softmax, "flashSparse_no_softmax", inputInfo, perf)
    naive_softmax = timing_decorator(flashSparse_naive_softmax, "flashSparse_naive_softmax", inputInfo, perf)
    stable_softmax = timing_decorator(flashSparse_stable_softmax, "flashSparse_stable_softmax", inputInfo, perf)
    
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
    f3s_1tb1rw = timing_decorator(TCFMM.f3s_1tb1rw, "f3s_1tb1rw", graphInfo, perf)
    f3s_1tb1rw_scheduled= timing_decorator(TCFMM.f3s_1tb1rw_scheduled, "f3s_1tb1rw_scheduled", graphInfo, perf)
    f3s_1tb1rw_scheduled_permuteV = timing_decorator(TCFMM.f3s_1tb1rw_scheduled_permuteV, "f3s_1tb1rw_scheduled_permuteV", graphInfo, perf)
    f3s_1tb1tcb = timing_decorator(TCFMM.f3s_1tb1tcb, "f3s_1tb1tcb", graphInfo, perf)
  else:
    f3s_1tb1rw = TCFMM.f3s_1tb1rw
    f3s_1tb1rw_scheduled = TCFMM.f3s_1tb1rw_scheduled
    f3s_1tb1rw_scheduled_permuteV = TCFMM.f3s_1tb1rw_scheduled_permuteV
    f3s_1tb1tcb = TCFMM.f3s_1tb1tcb

  if args.alg == 'f3s_1tb1tcb' or args.alg == 'all':
    RowWindowOffset, sortedRowWindows, TCblockRowid,\
    TCblocktileId, TCblockoffset, SparseAToXindex,\
    TBBoundaries, TCblockBitMap, block_count = f3s_preprocess_dataset(args, graphInfo, BLK_W=8)
    graphInfo.f3s_flop_and_lgd(RowWindowOffset, args.embedding_dim, use_1tb1tcb=True)
    apply_softmax = True
    save_sddmm_result = False
    f3s_1tb1tcb(
      RowWindowOffset, SparseAToXindex, TCblockBitMap, 
      graphInfo.num_nodes, Q, K, V, apply_softmax, save_sddmm_result)
    
  RowWindowOffset, sortedRowWindows, TCblockRowid,\
  TCblocktileId, TCblockoffset, SparseAToXindex,\
  TBBoundaries, TCblockBitMap, block_count = f3s_preprocess_dataset(args, graphInfo, BLK_W=16)
  graphInfo.f3s_flop_and_lgd(RowWindowOffset, args.embedding_dim, use_1tb1tcb=False)
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
    graphInfo.dfgnn_flop_and_lgd(args.embedding_dim)
    dfgnn_tiling = event_timing_decorator(GTConvFuse_inference_tiling, "dfgnn_tiling", graphInfo, perf)
    dfgnn_hyper = event_timing_decorator(GTConvFuse_inference_hyper, "dfgnn_hyper", graphInfo, perf)
  else:
    dfgnn_tiling = GTConvFuse_inference_tiling
    dfgnn_hyper = GTConvFuse_inference_hyper

  num_heads = 1
  Q = torch.rand(graphInfo.num_nodes, num_heads, args.embedding_dim, dtype=torch.float32, device=args.dev)
  K = torch.rand(graphInfo.num_nodes, num_heads, args.embedding_dim, dtype=torch.float32, device=args.dev)
  V = torch.rand(graphInfo.num_nodes, num_heads, args.embedding_dim, dtype=torch.float32, device=args.dev)
  row_pointers = graphInfo.get_row_pointers()
  column_index = graphInfo.get_column_index()
  num_edges = graphInfo.num_edges
  enable_dfgnn_hyper = graphInfo.enable_dfgnn_hyper()
  max_row_nnz = graphInfo.max_row_nnz
  if args.alg == "dfgnn_tiling" or args.alg == "all":
    smem_consume, val = dfgnn_preprocess_dataset(args, num_edges, max_row_nnz, alg="dfgnn_tiling")
    try:
      out = dfgnn_tiling(row_pointers, column_index, val, smem_consume, Q, K, V)
    except Exception as e:
      print(f"Error in dfgnn_tiling: {e}")
  if (args.alg == "dfgnn_hyper" or args.alg == "all") and enable_dfgnn_hyper:
    smem_consume, val = dfgnn_preprocess_dataset(args, num_edges, max_row_nnz, alg="dfgnn_hyper")
    rows = graphInfo.get_rows()
    try:
      out = dfgnn_hyper(row_pointers, column_index, rows, val, smem_consume, Q, K, V)
    except Exception as e:
      print(f"Error in dfgnn_hyper: {e}")

def dfgnn_preprocess_dataset(args, num_edges, max_row_nnz, alg):
  max_neigh = 128 # according to DF-GNN/DFGNN/layers/util.py
  WARP_SIZE = 32
  val = torch.ones(num_edges, dtype=torch.float32, device=args.dev)
  if alg == "dfgnn_tiling":
    smem_consume = (max_neigh + WARP_SIZE - 1) // WARP_SIZE * WARP_SIZE
    return smem_consume, val
  elif alg == "dfgnn_hyper":
    # smem_consume = (max_neigh * 8 + WARP_SIZE - 1) // WARP_SIZE * WARP_SIZE

    # each warp takes 1 row. 8 warps per block.
    # smem_consume = graphInfo.get_max_nnz_per_row() * 8
    smem_consume = max_row_nnz * 8
    return smem_consume, val
  else:
    raise ValueError(f"Invalid algorithm: {alg}")

# modified from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/transformer_conv.html#TransformerConv
# init function is modified to avoid allocating duplicate weights such as Q, K, V
class PygTransformerConv(torch_geometric.nn.conv.MessagePassing):
  def __init__(
    self,
    in_channels: Union[int, Tuple[int, int]],
    out_channels: int,
    heads: int = 1,
    concat: bool = True,
    beta: bool = False,
    dropout: float = 0.,
    edge_dim: Optional[int] = None,
    bias: bool = True,
    root_weight: bool = True,
    **kwargs,
):
    kwargs.setdefault('aggr', 'add')
    super().__init__(node_dim=0, **kwargs)

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.heads = heads
    self.beta = beta and root_weight
    self.root_weight = root_weight
    self.concat = concat
    self.dropout = dropout
    self.edge_dim = edge_dim
    self._alpha = None

    self.lin_edge = self.register_parameter('lin_edge', None)
    self.reset_parameters()

  @overload
  def forward(
      self,
      x: Union[torch.Tensor, PairTensor],
      edge_index: Adj,
      edge_attr: OptTensor = None,
      return_attention_weights: NoneType = None,
  ) -> torch.Tensor:
      pass
  @overload
  def forward(  # noqa: F811
      self,
      x: Union[torch.Tensor, PairTensor],
      edge_index: torch.Tensor,
      edge_attr: OptTensor = None,
      return_attention_weights: bool = None,
  ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
      pass
  @overload
  def forward(  # noqa: F811
      self,
      x: Union[torch.Tensor, PairTensor],
      edge_index: SparseTensor,
      edge_attr: OptTensor = None,
      return_attention_weights: bool = None,
  ) -> Tuple[torch.Tensor, SparseTensor]:
      pass

  def forward(  # noqa: F811
      self,
      x: Union[torch.Tensor, PairTensor],
      edge_index: Adj,
      edge_attr: OptTensor = None,
      return_attention_weights: Optional[bool] = None,
  ) -> Union[
          torch.Tensor,
          Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
          Tuple[torch.Tensor, SparseTensor],
  ]:
      r"""Runs the forward pass of the module.

      Args:
          x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
              features.
          edge_index (torch.Tensor or SparseTensor): The edge indices.
          edge_attr (torch.Tensor, optional): The edge features.
              (default: :obj:`None`)
          return_attention_weights (bool, optional): If set to :obj:`True`,
              will additionally return the tuple
              :obj:`(edge_index, attention_weights)`, holding the computed
              attention weights for each edge. (default: :obj:`None`)
      """
      H, C = self.heads, self.out_channels

      if isinstance(x, torch.Tensor):
          x = (x, x)

      query = self.lin_query(x[1]).view(-1, H, C)
      key = self.lin_key(x[0]).view(-1, H, C)
      value = self.lin_value(x[0]).view(-1, H, C)

      # propagate_type: (query: Tensor, key:Tensor, value: Tensor,
      #                  edge_attr: OptTensor)
      out = self.propagate(edge_index, query=query, key=key, value=value,
                            edge_attr=edge_attr)

      alpha = self._alpha
      self._alpha = None

      if self.concat:
          out = out.view(-1, self.heads * self.out_channels)
      else:
          out = out.mean(dim=1)

      if self.root_weight:
          x_r = self.lin_skip(x[1])
          if self.lin_beta is not None:
              beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
              beta = beta.sigmoid()
              out = beta * x_r + (1 - beta) * out
          else:
              out = out + x_r

      if isinstance(return_attention_weights, bool):
          assert alpha is not None
          if isinstance(edge_index, torch.Tensor):
              return out, (edge_index, alpha)
          elif isinstance(edge_index, SparseTensor):
              return out, edge_index.set_value(alpha, layout='coo')
      else:
          return out
        
  def message(self, query_i: torch.Tensor, key_j: torch.Tensor, value_j: torch.Tensor,
            edge_attr: Optional[torch.Tensor], index: torch.Tensor, ptr: Optional[torch.Tensor],
            size_i: Optional[int]) -> torch.Tensor:

    if self.lin_edge is not None:
        assert edge_attr is not None
        edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                  self.out_channels)
        key_j = key_j + edge_attr
    alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
    alpha = softmax(alpha, index, ptr, size_i)
    # self._alpha = alpha
    alpha = F.dropout(alpha, p=self.dropout, training=self.training)

    out = value_j
    if edge_attr is not None:
        out = out + edge_attr

    out = out * alpha.view(-1, self.heads, 1)
    return out

def pyg_gtconv(args, graphInfo, perf):
  pyg_layer = PygTransformerConv(args.embedding_dim, args.embedding_dim, heads=1, bias=False, root_weight=False)
  propagate = event_timing_decorator(pyg_layer.propagate, "pyg_gtconv", graphInfo, perf)
  num_nodes = graphInfo.num_nodes
  Q = torch.rand(num_nodes, 1, args.embedding_dim, dtype=torch.float32, device=args.dev)
  K = torch.rand(num_nodes, 1, args.embedding_dim, dtype=torch.float32, device=args.dev)
  V = torch.rand(num_nodes, 1, args.embedding_dim, dtype=torch.float32, device=args.dev)
  edge_index = graphInfo.get_edge_index().contiguous()
  try:
    out = propagate(edge_index=edge_index, query=Q, key=K, value=V, edge_attr=None)
  except Exception as e:
    print(f"Error in pyg_gtconv: {e}")

def route_methods(args, graphInfo, perf):
  if args.method == "flashSparse" or args.method == "all":
    # num_features and num_classes are for creating inputInfo.x and .y, which are not used in the kernel
    inputInfo = flashSparse_preprocess_dataset(args, graphInfo)
    inputInfo.flashSparse_flop_and_lgd(inputInfo.row_pointers, args.embedding_dim)
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
    test_algs = [args.alg]
  perf = Perf(test_algs, test_dataset)
  for dataset in test_dataset:
    graphInfo = load_dataset(dataset)
    check_gpu_memory()
    route_methods(args, graphInfo, perf)
  perf.runtime_pd.to_csv(f"baseline_comp_kernel_only_runtime_{args.method}_{args.alg}_{args.dataset}_{get_gpu_model()}.csv")
  perf.throughput_pd.to_csv(f"baseline_comp_kernel_only_throughput_{args.method}_{args.alg}_{args.dataset}_{get_gpu_model()}.csv")
  perf.bandwidth_pd.to_csv(f"baseline_comp_kernel_only_bandwidth_{args.method}_{args.alg}_{args.dataset}_{get_gpu_model()}.csv")
  
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
