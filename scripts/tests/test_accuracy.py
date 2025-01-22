import numpy as np
import scipy as sp
import torch
import TCFMM
import time
import torch.nn.functional as F

def convert_row_major_nz_to_block_row_major(A, BLK_H, BLK_W):
  nnz = torch.empty(0, dtype= A.dtype, device=A.device)
  n_block_per_row = (A.shape[1] + BLK_W - 1) // BLK_W
  n_block_per_col = (A.shape[0] + BLK_H - 1) // BLK_H
  for i in range(n_block_per_col):
    for j in range(n_block_per_row):
      row = i * BLK_H
      col = j * BLK_W
      block = A[row:row+BLK_H, col:col+BLK_W]
      block = block[block!=0]
      nnz = torch.cat((nnz, block))
  return nnz

def main():
  n_runs = 1
  n_test = 1
  BLK_H = 16
  BLK_W = 8
  n_heads = 1
  feature_size = 64
 
  size = 128
  density = 0.5

  half_v_float_edge_atten = []
  half_v_float_final = []
  edge_atten_err_ftc_fp16_v_true_fp32 = []
  edge_atten_err_ftc_fp32_v_true_fp32 = []
  final_err_ftc_fp16_v_true_fp32 = []
  final_err_ftc_fp32_v_true_fp32 = []
  fusedRs = []
  # np.random.seed(26)
  # torch.manual_seed(26)
  for n in range(n_test):
    np.random.seed(n)
    torch.manual_seed(n)

    print(f"----------{n}-----------")
 
    A_csr_h = sp.sparse.random(size, size, density=density, format='csr', data_rvs=np.random.rand)
    A_csr_h = (A_csr_h + A_csr_h.T) / 2
    # Round and convert all non-zero entries to 1
    A_csr_h.data = np.ceil(A_csr_h.data, dtype=np.float32)
    A_dense = torch.tensor(A_csr_h.todense()).cuda()
    A_dense_half = A_dense.to(torch.float16)

    # attention weights
    # attention_w = torch.randn(1, n_heads, device='cuda')
    attention_w = torch.ones(1, n_heads, device='cuda')
    # generate the dense feature matrix
    X = torch.rand(size, feature_size, dtype=torch.float32, device='cuda')
    # pad the feature matrix to have a multiple of 16 columns
    if feature_size % 16 != 0:
      padding_len = 16 - feature_size % 16
      X = F.pad(X, (0, padding_len, 0, 0))
      print(X.shape)
    X_half = X.to(torch.float16)
    
    sddmm_half = (X_half @ X_half.T) * A_dense_half * attention_w.half()
    sddmm = (X @ X.T) * A_dense * attention_w
    edge_attention_true_half = convert_row_major_nz_to_block_row_major(sddmm_half, BLK_H, BLK_W) 
    edge_attention_true = convert_row_major_nz_to_block_row_major(sddmm, BLK_H, BLK_W)
    edge_attention_true_norm = torch.norm(edge_attention_true)
    rel_err = torch.norm(edge_attention_true - edge_attention_true_half)/edge_attention_true_norm
    half_v_float_edge_atten.append(rel_err.item())
    true = sddmm @ X
    true_half = sddmm_half @ X_half
    true_norm = torch.norm(true)
    rel_err = torch.norm(true - true_half)/true_norm
    half_v_float_final.append(rel_err.item())


    use_f32_edge_attention = False
    save_edge_attention = True
    num_row_windows = (size + BLK_H - 1) // BLK_H
    edgeToColumn = torch.zeros(A_csr_h.nnz, dtype=torch.int)
    edgeToRow = torch.zeros(A_csr_h.nnz, dtype=torch.int)
    blockPartition = torch.zeros(num_row_windows, dtype=torch.int)
    blockPartition_cuda  = blockPartition.cuda()
    edgeToColumn_cuda = edgeToColumn.cuda()
    edgeToRow_cuda  = edgeToRow.cuda()
    RowWindowOffset, sortedRowWindows, TCblockRowid,\
        TCblocktileId, TCblockoffset, SparseAToXindex,\
            block_count = TCFMM.preprocess_gpu(torch.IntTensor(A_csr_h.indices).cuda(), torch.IntTensor(A_csr_h.indptr).cuda(), size, BLK_H, BLK_W, blockPartition_cuda, edgeToColumn_cuda, edgeToRow_cuda)
    # debug
    print(TCblockoffset[1])
    cols = SparseAToXindex[:5]
    print(X_half[cols, :5].transpose(0, 1))
    # start_time = time.time()
    # for i in range(n_runs):
    #   fusedR, edge_attention = TCFMM.fusedMM_forward(X, RowWindowOffset, TCblocktileId, TCblockoffset, SparseAToXindex, size, save_edge_attention, use_f32_edge_attention)
    # FTC_half_inter_time = (time.time() - start_time)/n_runs
    # print(f"FTC_half_inter_time: {FTC_half_inter_time}")
    # rel_err = torch.norm(edge_attention - edge_attention_true) / edge_attention_true_norm
    # edge_atten_err_ftc_fp16_v_true_fp32.append(rel_err.item())
    # rel_err = torch.norm(fusedR - true) / true_norm
    # final_err_ftc_fp16_v_true_fp32.append(rel_err.item())

    use_f32_edge_attention = True
    use_m8n32k16 = False
    start_time = time.time()
    for i in range(n_runs):
      fusedR, edge_attention = TCFMM.fusedMM_forward(X_half, RowWindowOffset, TCblocktileId, TCblockoffset, SparseAToXindex, size, save_edge_attention, use_f32_edge_attention, use_m8n32k16)
      print(fusedR.shape)
    FTC_f32_inter_time = (time.time() - start_time)/n_runs
    print(f"FTC_f32_inter_time: {FTC_f32_inter_time}")
    rel_err = torch.norm(edge_attention - edge_attention_true) / edge_attention_true_norm
    print(edge_attention[0:20])
    print(edge_attention_true[0:20])
    edge_atten_err_ftc_fp32_v_true_fp32.append(rel_err.item())
    rel_err = torch.norm(fusedR - true) / true_norm
    final_err_ftc_fp32_v_true_fp32.append(rel_err.item())
    

  half_v_float_edge_atten_mean = np.mean(np.array(half_v_float_edge_atten))
  half_v_float_final_mean = np.mean(np.array(half_v_float_final))
  # mean_edge_atten_err_ftc_fp16_v_true_fp32 = np.mean(np.array(edge_atten_err_ftc_fp16_v_true_fp32))
  # mean_final_err_ftc_fp16_v_true_fp32 = np.mean(np.array(final_err_ftc_fp16_v_true_fp32))
  mean_edge_atten_err_ftc_fp32_v_true_fp32 = np.mean(np.array(edge_atten_err_ftc_fp32_v_true_fp32))
  mean_final_err_ftc_fp32_v_true_fp32 = np.mean(np.array(final_err_ftc_fp32_v_true_fp32))

  print(f"edge attention pytorch half vs single: {half_v_float_edge_atten_mean}")
  print(f"final solution pytorch half vs float: {half_v_float_final_mean}")
  # print(f"FTCMM (fp16 SDDMM result) edge attention vs pytorch float: {mean_edge_atten_err_ftc_fp16_v_true_fp32}")
  # print(f"FTCMM (fp16 SDDMM result) final solution vs pytorch float: {mean_final_err_ftc_fp16_v_true_fp32}")
  print(f"FTCMM edge attention vs pytorch float: {mean_edge_atten_err_ftc_fp32_v_true_fp32}")
  print(f"FTCMM final solution vs pytorch float: {mean_final_err_ftc_fp32_v_true_fp32}")

if __name__ == "__main__":
  main()

