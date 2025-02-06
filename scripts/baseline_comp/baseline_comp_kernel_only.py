import argparse
import torch
import numpy as np
from scipy.sparse import coo_matrix
from torch_geometric.utils import softmax
import FS_Block
import FS_SDDMM
import FS_SpMM


# only for flashSparse
class InputInfo:
  def __init__(self):
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

def timing_decorator(kernel):
  def wrapper(*args, **kwargs):
    niter = 10
    # warmup
    for i in range(5):
      out = kernel(*args, **kwargs)
    # torch.cuda.synchronize()
    print("warmup done")
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
    print(f"average execution time: {execution_time} ms")
    return out
  return wrapper


def flashSparse_no_softmax(X_prime, inputInfo):
  att = FS_SDDMM.forward_gen_fp16_gnn(   
            X_prime.size(1),                                      
            inputInfo.row_pointers, 
            inputInfo.column_index, 
            inputInfo.degrees, 
            inputInfo.t_window_rowTensor,
            X_prime,X_prime,inputInfo.max)[0] 
  h_prime = FS_SpMM.forward_fp16_gnn(   
              inputInfo.row_pointers, 
              inputInfo.column_index, 
              att, 
              inputInfo.t_window_rowTensor,
              inputInfo.t_atomicTensor,
              X_prime, 
              inputInfo.num_nodes, 
              X_prime.size(1), 
              inputInfo.num_nodes_ori)[0]
  return h_prime

def flashSparse_naive_softmax(X_prime, inputInfo):
  att = FS_SDDMM.forward_gen_fp16_gnn(   
              X_prime.size(1),                                      
              inputInfo.row_pointers, 
              inputInfo.column_index, 
              inputInfo.degrees, 
              inputInfo.t_window_rowTensor,
              X_prime,X_prime,inputInfo.max)[0] 
  att = torch.exp(att) # softmax
  rows_sum = FS_SpMM.forward_fp16_gnn_ones(   
              inputInfo.row_pointers, 
              inputInfo.column_index, 
              att, 
              inputInfo.t_window_rowTensor,
              inputInfo.t_atomicTensor,
              inputInfo.ones, 
              inputInfo.num_nodes, 
              inputInfo.ones.size(1), 
              inputInfo.num_nodes_ori)[0]
  h_prime = FS_SpMM.forward_fp16_gnn(   
              inputInfo.row_pointers, 
              inputInfo.column_index, 
              att, 
              inputInfo.t_window_rowTensor,
              inputInfo.t_atomicTensor,
              X_prime, 
              inputInfo.num_nodes, 
              X_prime.size(1), 
              inputInfo.num_nodes_ori)[0].half()
  h_prime = h_prime.div(rows_sum) # softmax

def flashSparse_stable_softmax(X_prime, inputInfo, edge_att_rand):
  att = FS_SDDMM.forward_gen_fp16_gnn(   
              X_prime.size(1),                                      
              inputInfo.row_pointers, 
              inputInfo.column_index, 
              inputInfo.degrees, 
              inputInfo.t_window_rowTensor,
              X_prime,X_prime,inputInfo.max)[0]
  softmax(edge_att_rand, ptr=inputInfo.orig_row_pointers.to('cuda'))
  h_prime = FS_SpMM.forward_fp16_gnn(   
              inputInfo.row_pointers, 
              inputInfo.column_index, 
              att, 
              inputInfo.t_window_rowTensor,
              inputInfo.t_atomicTensor,
              X_prime, 
              inputInfo.num_nodes, 
              X_prime.size(1), 
              inputInfo.num_nodes_ori)[0].half()

def route_flashSparse(args, inputInfo, Q):
  if args.use_cuda_event:
    no_softmax = timing_decorator(flashSparse_no_softmax)
    naive_softmax = timing_decorator(flashSparse_naive_softmax)
    stable_softmax = timing_decorator(flashSparse_stable_softmax)
  else:
    no_softmax = flashSparse_no_softmax
    naive_softmax = flashSparse_naive_softmax
    stable_softmax = flashSparse_stable_softmax
  if(args.alg == "fs_no_softmax"):
    no_softmax(Q, inputInfo)
  elif(args.alg == "fs_naive_softmax"):
    naive_softmax(Q, inputInfo)
  elif(args.alg == "fs_stable_softmax"):
    edge_att_rand = torch.rand(inputInfo.num_edges, dtype=torch.float16, device=args.dev)
    stable_softmax(Q, inputInfo, edge_att_rand)
  else:
    raise ValueError(f"Invalid algorithm: {args.alg}")

def flashSparse_preprocess_dataset(args, orig_row_pointers, orig_column_index, num_nodes, num_edges):
  partSize = 32
  window = 8
  wide = 16
  inputInfo = InputInfo()
  print("starting blockProcess_sddmm_balance_gnn")
  inputInfo.orig_row_pointers = orig_row_pointers
  inputInfo.row_pointers, inputInfo.column_index, \
  inputInfo.degrees, inputInfo.t_window_rowTensor, \
  inputInfo.t_atomicTensor = FS_Block.blockProcess_sddmm_balance_gnn(orig_row_pointers.cpu(), orig_column_index.cpu(), window, wide, partSize)
  inputInfo.row_pointers = inputInfo.row_pointers.cuda()
  inputInfo.column_index = inputInfo.column_index.cuda()
  inputInfo.degrees = inputInfo.degrees.cuda()
  inputInfo.t_window_rowTensor = inputInfo.t_window_rowTensor.cuda()
  inputInfo.t_atomicTensor = inputInfo.t_atomicTensor.cuda()
  print("finished blockProcess_sddmm_balance_gnn")
  inputInfo.num_nodes_ori = num_nodes
  if num_nodes%16 !=0 :
    inputInfo.num_nodes = num_nodes + 16 - num_nodes%16
  else:
    inputInfo.num_nodes = num_nodes
  inputInfo.num_edges = num_edges
  inputInfo.ones = torch.ones((inputInfo.num_nodes_ori,1), dtype=torch.float16, device=args.dev)
  max_vectors = torch.max(inputInfo.row_pointers[1:]- inputInfo.row_pointers[:-1])
  if max_vectors%wide > 0 :
      max_vectors += (wide - (max_vectors%wide))
  inputInfo.max = max_vectors / wide
  
  if inputInfo.max % 4 > 0 :
      inputInfo.max += 4 - inputInfo.max%4
  return inputInfo

def route_f3s(args, sparse_rep, Q, K, V, num_nodes):
  import TCFMM
  if args.use_cuda_event:
    f3s_1tb1rw = timing_decorator(TCFMM.f3s_1tb1rw)
    f3s_1tb1rw_scheduled= timing_decorator(TCFMM.f3s_1tb1rw_scheduled)
    f3s_1tb1tcb = timing_decorator(TCFMM.f3s_1tb1tcb)
  else:
    f3s_1tb1rw = TCFMM.f3s_1tb1rw
    f3s_1tb1rw_scheduled = TCFMM.f3s_1tb1rw_scheduled
    f3s_1tb1tcb = TCFMM.f3s_1tb1tcb
  RowWindowOffset, sortedRowWindows, TCblockRowid,\
  TCblocktileId, TCblockoffset, SparseAToXindex,\
  TBBoundaries, TCblockBitMap, block_count = sparse_rep
  if args.alg == 'f3s_1tb1rw':
    final_result, sddmm_result_1tb1rw = f3s_1tb1rw(
      RowWindowOffset, SparseAToXindex, TCblockBitMap, 
      num_nodes, Q, K, V, args.n_warp_per_block, True)
  elif args.alg == 'f3s_1tb1rw_no_softmax':
    final_result, sddmm_result_1tb1rw = f3s_1tb1rw(
      RowWindowOffset, SparseAToXindex, TCblockBitMap, 
      num_nodes, Q, K, V, args.n_warp_per_block, False)
  elif args.alg == 'f3s_1tb1rw_scheduled':
    final_result, sddmm_result_1tb1rw_scheduled = f3s_1tb1rw_scheduled(
      RowWindowOffset, sortedRowWindows, SparseAToXindex, TCblockBitMap, 
      num_nodes, Q, K, V, args.n_warp_per_block)
  elif args.alg == 'f3s_1tb1tcb':
    apply_softmax = True
    save_sddmm_result = False
    final_result, sddmm_result_1tb1tcb = f3s_1tb1tcb(
      RowWindowOffset, SparseAToXindex, TCblockBitMap, 
      num_nodes, Q, K, V, apply_softmax, save_sddmm_result)
  else:
    raise ValueError(f"Invalid algorithm: {args.alg}")

def f3s_preprocess_dataset(args, indptr, indices, num_nodes, num_edges):
  from TCFMM import preprocess_gpu
  BLK_H = 16
  if args.alg == 'f3s_1tb1rw' or args.alg == 'f3s_1tb1rw_scheduled':
    BLK_W = 16
  else:
    BLK_W = 8
  # Set up tensors for preprocessing
  num_row_windows = (num_nodes + BLK_H - 1) // BLK_H
  # Move tensors to GPU
  blockPartition_cuda = torch.zeros(num_row_windows, dtype=torch.int).cuda()
  edgeToColumn_cuda = torch.zeros(num_edges, dtype=torch.int).cuda()
  edgeToRow_cuda = torch.zeros(num_edges, dtype=torch.int).cuda()
  return preprocess_gpu(indices, indptr, num_nodes, 
                              BLK_H, BLK_W, blockPartition_cuda, 
                              edgeToColumn_cuda, edgeToRow_cuda)

def route_dfgnn(args, row_pointers, column_index, rows, smem_consume, val, Q, K, V):
  from DFGNN.operators.fused_gtconv import GTConvFuse_inference_tiling, GTConvFuse_inference_hyper
  if args.use_cuda_event:
    dfgnn_tiling = timing_decorator(GTConvFuse_inference_tiling)
    dfgnn_hyper = timing_decorator(GTConvFuse_inference_hyper)
  else:
    dfgnn_tiling = GTConvFuse_inference_tiling
    dfgnn_hyper = GTConvFuse_inference_hyper
  if args.alg == "dfgnn_tiling":
    out = dfgnn_tiling(row_pointers, column_index, val, smem_consume, Q, K, V)
  elif args.alg == "dfgnn_hyper":
    out = dfgnn_hyper(row_pointers, column_index, rows, val, smem_consume, Q, K, V)
  else:
    raise ValueError(f"Invalid algorithm: {args.alg}")

def dfgnn_preprocess_dataset(args, indices, rows, Q, K, V):
  # add 1 dimension of 1 to Q, K, V
  Q = Q.unsqueeze(1).float()
  K = K.unsqueeze(1).float()
  V = V.unsqueeze(1).float()
  max_neigh = 128 # according to DF-GNN/DFGNN/layers/util.py
  WARP_SIZE = 32
  assert len(rows) == len(indices)
  val = torch.ones(len(rows), dtype=torch.float32, device=args.dev)
  if args.alg == "dfgnn_tiling":
    smem_consume = (max_neigh + WARP_SIZE - 1) // WARP_SIZE * WARP_SIZE
    return smem_consume, val, Q, K, V
  elif args.alg == "dfgnn_hyper":
    smem_consume = (max_neigh * 8 + WARP_SIZE - 1) // WARP_SIZE * WARP_SIZE
    return smem_consume, val, Q, K, V
  
def route_methods(args, row_pointers, column_index, rows, num_nodes, num_edges, Q, K, V):
  # load dataset
  if args.method == "f3s":
    sparse_rep = f3s_preprocess_dataset(args, row_pointers, column_index, num_nodes, num_edges)
    route_f3s(args, sparse_rep, Q, K, V, num_nodes)
  elif args.method == "flashSparse":
    # num_features and num_classes are for creating inputInfo.x and .y, which are not used in the kernel
    inputInfo = flashSparse_preprocess_dataset(args, row_pointers, column_index, num_nodes, num_edges)
    print(f"inputInfo.num_edges: {inputInfo.num_edges}, inputInfo.num_nodes: {inputInfo.num_nodes}, inputInfo.max: {inputInfo.max}")
    print(f"inputInfo.row_pointers.device: {inputInfo.row_pointers.device}")
    route_flashSparse(args, inputInfo, Q)
  elif args.method == "df-gnn":
    smem_consume, val, Q, K, V = dfgnn_preprocess_dataset(args, column_index, rows, Q, K, V)
    route_dfgnn(args, row_pointers, column_index, rows, smem_consume, val, Q, K, V)
  else:
    raise ValueError(f"Invalid method: {args.method}")

def load_dataset(args):
  path = f"/share/crsp/lab/amowli/share/Fused3S/dataset/{args.dataset}.npz"
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
  # row indices of all non-zero elements in the adjacency matrix, required by dfgnn
  row_indices = np.repeat(np.arange(adj.shape[0]), np.diff(adj.indptr))
  rows = torch.IntTensor(row_indices).cuda()
  return row_pointers, column_index, rows, num_nodes, num_edges

def main(args):
  num_heads = 1
  args.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  row_pointers, column_index, rows, num_nodes, num_edges = load_dataset(args)
  print(f"dataset: {args.dataset}, num_nodes: {num_nodes}, num_edges: {num_edges}")
  Q = torch.rand(num_nodes, args.embedding_dim, dtype=torch.float16, device=args.dev)
  K = torch.rand(num_nodes, args.embedding_dim, dtype=torch.float16, device=args.dev)
  V = torch.rand(num_nodes, args.embedding_dim, dtype=torch.float16, device=args.dev)
  route_methods(args, row_pointers, column_index, rows, num_nodes, num_edges, Q, K, V)
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', '-emb', type=int, default=128,
                       help='Dimension of node embeddings')
    parser.add_argument('--n_warp_per_block', '-nw', type=int, default=8,
                       help='Number of warps per block')
    parser.add_argument('--dataset', '-d', type=str, default="reddit",
                       choices=["reddit", "ppa", "protein", "cora", "pubmed"])
    parser.add_argument('--method', '-m', type=str, default="f3s",
                       choices=["f3s", "flashSparse", "df-gnn"])
    parser.add_argument("--alg", '-a', type=str, default='f3s_1tb1rw_scheduled', 
                        choices=['f3s_1tb1tcb', 'f3s_1tb1rw', 'f3s_1tb1rw_no_softmax', 'f3s_1tb1rw_scheduled', 
                                 'fs_no_softmax', 'fs_naive_softmax', 'fs_stable_softmax', 
                                 'dfgnn_tiling', 'dfgnn_hyper'])
    parser.add_argument("--use_cuda_event", action='store_true', 
                        help='Use CUDA event to measure time, no timer is used otherwise')
    args = parser.parse_args()
    main(args)
