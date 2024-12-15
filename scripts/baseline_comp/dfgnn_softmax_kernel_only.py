import torch
from DFGNN.layers.util import preprocess_softmax
from DFGNN.operators.fused_gtconv import GTConvFuse_inference_softmax
from dgl.data import (CoraGraphDataset, RedditDataset, PubmedGraphDataset)
# from dgl.data.lrgb import RedditDataset
# import TCFMM

def run_dfgnn(indptr, indices, rows, val, smem_consume, Q, K, V):
  out = GTConvFuse_inference_softmax(indptr, indices, rows, val, smem_consume, Q, K, V)

# def run_f3s(indptr, indices, num_nodes, num_edges, Q, K, V, BLK_H, BLK_W):
#   # Set up tensors for preprocessing
#   num_row_windows = (num_nodes + BLK_H - 1) // BLK_H
#   # Move tensors to GPU
#   blockPartition_cuda = torch.zeros(num_row_windows, dtype=torch.int).cuda()
#   edgeToColumn_cuda = torch.zeros(num_edges, dtype=torch.int).cuda()
#   edgeToRow_cuda = torch.zeros(num_edges, dtype=torch.int).cuda()
#   RowWindowOffset, TCblockRowid, TCblocktileId,\
#   TCblockoffset, SparseAToXindex, TCblockBitMap,\
#   block_count = TCFMM.preprocess_gpu(indices, indptr, num_nodes, BLK_H, BLK_W, blockPartition_cuda, edgeToColumn_cuda, edgeToRow_cuda)
#   apply_softmax = True
#   save_sddmm_result = False
#   TCFMM.f3S_forward(RowWindowOffset, SparseAToXindex, TCblockBitMap, num_nodes, Q, K, V, apply_softmax, save_sddmm_result)

def main():
  # BLK_H = 16
  # BLK_W = 8
  embedding_dim = 320
  num_heads = 1
  dataset_dir = "/share/crsp/lab/amowli/share/Fused3S/dfgnn/"
  dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  dataset = PubmedGraphDataset(raw_dir=dataset_dir)
  g = dataset[0].to(dev)
  num_nodes = g.num_nodes()
  num_edges = g.num_edges()
  print(g)
  print(g.ndata["feat"].shape)
  params = preprocess_softmax(g)
  del g  
  torch.cuda.empty_cache()
  indptr, indices, rows, val, smem_consume = params
  Q = torch.rand(num_nodes, num_heads, embedding_dim, dtype=torch.float32, device=dev)
  K = torch.rand(num_nodes, num_heads, embedding_dim, dtype=torch.float32, device=dev)
  V = torch.rand(num_nodes, num_heads, embedding_dim, dtype=torch.float32, device=dev)
  run_dfgnn(indptr, indices, rows, val, smem_consume, Q, K, V)
  # torch.cuda.empty_cache()

  # if(num_heads == 1):
  #   Q = Q.reshape(num_nodes, -1).to(torch.float16)
  #   K = K.reshape(num_nodes, -1).to(torch.float16)
  #   V = V.reshape(num_nodes, -1).to(torch.float16)
  # else:
  #   raise Exception("num_heads must be 1 for now")
  # run_f3s(indptr, indices, num_nodes, num_edges, Q, K, V, BLK_H, BLK_W)
  

if __name__ == "__main__":
    main()
