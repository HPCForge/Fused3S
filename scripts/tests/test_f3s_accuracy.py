import numpy as np
import scipy as sp
import torch
import F3S
import time
import torch.nn.functional as F
import argparse
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

# mat is a 2D pytorch tensor. this fucntion does the following:
# 1. divided it into row blocks of height BLK_H. If the last row block is not full, pad it with zeros.
# 2. Inside each row block, remove all the columns that are all 0s.
# 3. Inside each row block, group the remaining columns into groups of BLK_W. This should result in submatrices of size BLK_H x BLK_W
# 4. Go through these submatrices in each row block. for each submatrices, write their values into R in row-major order.
# 5. return R
def process_matrix(mat, BLK_H, BLK_W, use_1tb1rw):
    M, K = mat.shape
    num_blocks = (M + BLK_H - 1) // BLK_H  # Ceiling division to get the number of row blocks

    submatrices_list = []

    for i in range(num_blocks):
        # Step 1: Divide into row blocks of height BLK_H
        block_start_row = i * BLK_H
        block_end_row = min((i + 1) * BLK_H, M)
        block_rows = mat[block_start_row:block_end_row, :]
        block_height = block_rows.shape[0]

        # Step 2: Remove columns that are all zeros within this block
        non_zero_columns = (block_rows != 0).any(dim=0)
        block_rows = block_rows[:, non_zero_columns]

        # Step 3: Group remaining columns into groups of BLK_W
        num_columns_remaining = block_rows.shape[1]
        num_tcb_in_rw = (num_columns_remaining + BLK_W - 1) // BLK_W  # Ceiling division
        if use_1tb1rw:
          # Round up to nearest even number 
          # this is because in 1tb1rw we need 16x16 blocks in SDDMM result for SpMM
          num_tcb_in_rw = (num_tcb_in_rw + 1) // 2 * 2
        padded_num_columns = num_tcb_in_rw * BLK_W
        pad_columns = padded_num_columns - num_columns_remaining

        # Pad columns with zeros if necessary
        if pad_columns > 0:
            padding = torch.zeros((block_rows.shape[0], pad_columns), dtype=block_rows.dtype, device=block_rows.device)
            block_rows = torch.cat([block_rows, padding], dim=1)

        # Pad rows with zeros if necessary to ensure each block has BLK_H rows
        if block_height < BLK_H:
            pad_rows = BLK_H - block_height
            padding = torch.zeros((pad_rows, block_rows.shape[1]), dtype=block_rows.dtype, device=block_rows.device)
            block_rows = torch.cat([block_rows, padding], dim=0)

        # Reshape and permute to get submatrices of size BLK_H x BLK_W
        block_rows = block_rows.view(BLK_H, num_tcb_in_rw, BLK_W)
        block_submatrices = block_rows.permute(1, 0, 2)  # Shape: (num_tcb_in_rw, BLK_H, BLK_W)

        submatrices_list.append(block_submatrices.flatten())

    # Step 4: Concatenate all submatrices into R
    R = torch.cat(submatrices_list, dim=0)  # Shape: (N, BLK_H, BLK_W)

    return R

def pad_csr_matrix(mat, desired_rows, desired_cols):
    current_rows, current_cols = mat.shape
    
    # Pad columns if needed
    if current_cols < desired_cols:
        padding_cols = sp.sparse.csr_matrix((current_rows, desired_cols - current_cols))
        mat = sp.sparse.hstack([mat, padding_cols])
    
    # Pad rows if needed
    if current_rows < desired_rows:
        padding_rows = sp.sparse.csr_matrix((desired_rows - current_rows, desired_cols))
        mat = sp.sparse.vstack([mat, padding_rows])
    
    return mat

# compute the softmax of the matrix, 
# but only for elements whose corresponding value in A is not 0
def row_wise_softmax_mask(matrix, A):
    mask = (A != 0)
    neg_inf = torch.tensor(float('-inf'), device=matrix.device, dtype=matrix.dtype)
    one_val = torch.tensor(1.0, device=matrix.device, dtype=matrix.dtype)
    
    masked_matrix = torch.where(mask, matrix, neg_inf)
    row_max, _ = torch.max(masked_matrix, dim=1, keepdim=True)
    shifted_matrix = torch.where(mask, matrix - row_max, neg_inf)
    exp_matrix = torch.exp(shifted_matrix)
    row_sum = exp_matrix.sum(dim=1, keepdim=True)
    # Prevent division by zero if a row is entirely masked
    row_sum = torch.where(row_sum == 0, one_val, row_sum)
    softmax_matrix = exp_matrix / row_sum
    return softmax_matrix

def main(args):
  n_runs = 1
  n_test = 1
  BLK_H = 16
  BLK_W = 8
  # whether to use new parallel strategy where each warp computes a tcb
  apply_softmax = not args.skip_softmax
  embedding_size = args.embedding_size
  size = args.size
  density = args.density
  use_1tb1rw = False
  if args.alg == '1tb1rw' or args.alg == '1tb1rw_scheduled' or args.alg == '1tb1rw_scheduled_permuteV':
    use_1tb1rw = True
    BLK_W = 16
  save_sddmm_result = True
  # for 1tb1rw and 1tbnrw, the number of warps per block
  nWarpPerBlock = 8
  half_v_float_sddmm = []
  half_v_float_final = []
  f3s_v_true_fp32_sddmm = []
  f3s_v_true_fp32_final = []
  # np.random.seed(26)
  # torch.manual_seed(26)
  for n in range(n_test):
    np.random.seed(n)
    torch.manual_seed(n)
 
    A_csr_h = sp.sparse.random(size, size, density=density, format='csr', data_rvs=np.random.rand)

    # Round and convert all non-zero entries to 1
    A_csr_h.data = np.ceil(A_csr_h.data, dtype=np.float32)
    A_dense = torch.tensor(A_csr_h.todense()).cuda()
    A_dense_half = A_dense.to(torch.float16)

    # generate the dense feature matrix
    Q = torch.rand(size, embedding_size, dtype=torch.float32, device='cuda')
    K = torch.rand(size, embedding_size, dtype=torch.float32, device='cuda')
    V = torch.rand(size, embedding_size, dtype=torch.float32, device='cuda')
    # pad the feature matrix to have a multiple of 16 columns
    # if embedding_size % BLK_H != 0:
    col_padding_len = 0 if embedding_size % BLK_H == 0 else BLK_H - embedding_size % BLK_H
    # row_padding_len = 0 if size % BLK_H == 0 else BLK_H - size % BLK_H
    row_padding_len = 0
    Q = F.pad(Q, (0, col_padding_len, 0, row_padding_len), "constant", 0)
    K = F.pad(K, (0, col_padding_len, 0, row_padding_len), "constant", 0)
    V = F.pad(V, (0, col_padding_len, 0, row_padding_len), "constant", 0)
    Q_half = Q.to(torch.float16)
    print(f"Q_half.shape: {Q_half.shape}")
    K_half = K.to(torch.float16)
    print(f"K_half.shape: {K_half.shape}")
    V_half = V.to(torch.float16)
    print(f"V_half.shape: {V_half.shape}")
    K_half_cpu = K_half.to("cpu")
    
    sddmm_half_og_form = (Q_half @ K_half.T) * A_dense_half
    sddmm_og_form = (Q @ K.T) * A_dense
    # BLK_M = 16, BLK_N = 8, but BLK_H = 16, BLK_W = 16
    # I'm detaching the size of the output block from the mma block size.
    # This is to deal with odd number of TCBs.
    sddmm_true_half = process_matrix(sddmm_half_og_form, 16, 8, use_1tb1rw)
    sddmm_true = process_matrix(sddmm_og_form, 16, 8, use_1tb1rw)
    sddmm_true_norm = torch.norm(sddmm_true)
    rel_err = torch.norm(sddmm_true - sddmm_true_half)/sddmm_true_norm
    half_v_float_sddmm.append(rel_err.item())
    if apply_softmax:
      softmax_true_half_og_form = row_wise_softmax_mask(sddmm_half_og_form, A_dense_half)
      softmax_true_og_form = row_wise_softmax_mask(sddmm_og_form, A_dense)
    S = softmax_true_og_form if apply_softmax else sddmm_og_form
    S_half = softmax_true_half_og_form if apply_softmax else sddmm_half_og_form
    true = S @ V
    true_half = S_half @ V_half
    true_norm = torch.norm(true)
    rel_err = torch.norm(true - true_half)/true_norm
    half_v_float_final.append(rel_err.item())

    
    num_row_windows = (size + BLK_H - 1) // BLK_H
    edgeToColumn = torch.zeros(A_csr_h.nnz, dtype=torch.int)
    edgeToRow = torch.zeros(A_csr_h.nnz, dtype=torch.int)
    blockPartition = torch.zeros(num_row_windows, dtype=torch.int)
    blockPartition_cuda  = blockPartition.cuda()
    edgeToColumn_cuda = edgeToColumn.cuda()
    edgeToRow_cuda  = edgeToRow.cuda()
    indices = torch.IntTensor(A_csr_h.indices).cuda()
    indptr = torch.IntTensor(A_csr_h.indptr).cuda()
    RowWindowOffset, sortedRowWindows, TCblockRowid,_, _,\
    SparseAToXindex, TBBoundaries, TCblockBitMap, _ = F3S.preprocess_gpu(indices, indptr, size, 
                                                                  BLK_H, BLK_W, 
                                                                  blockPartition_cuda, 
                                                                  edgeToColumn_cuda, 
                                                                  edgeToRow_cuda)
    for i in range(n_runs):
      if args.alg == '1tb1rw':
        check_sm_active_time = False
        print("using 1tb1rw")
        time, fusedR, sddmm_result = F3S.f3s_1tb1rw(RowWindowOffset, SparseAToXindex, TCblockBitMap, 
                                                size, Q_half, K_half, V_half, nWarpPerBlock, apply_softmax, check_sm_active_time)
      elif args.alg == '1tb1rw_scheduled':
        print("using 1tb1rw_scheduled")
        check_sm_active_time = False
        time, fusedR, sddmm_result = F3S.f3s_1tb1rw_scheduled(RowWindowOffset, sortedRowWindows, SparseAToXindex, TCblockBitMap, 
                                                          size, Q_half, K_half, V_half, nWarpPerBlock, check_sm_active_time)
      elif args.alg == '1tb1rw_scheduled_permuteV':
        print("using 1tb1rw_scheduled_permuteV")
        time, fusedR, sddmm_result = F3S.f3s_1tb1rw_scheduled_permuteV(RowWindowOffset, sortedRowWindows, SparseAToXindex, TCblockBitMap, 
                                                          size, Q_half, K_half, V_half, nWarpPerBlock)
      elif args.alg == '1tbnrw':
        print("using 1tbnrw")
        time, sddmm_result = F3S.sddmm_1tbnrw(RowWindowOffset, TBBoundaries, TCblockRowid, SparseAToXindex, TCblockBitMap, 
                                          size, Q_half, K_half, nWarpPerBlock)[0]
        fusedR = None
      elif args.alg == '1tb1tcb':
        print("using 1tb1tcb")
        time, fusedR, sddmm_result = F3S.f3s_1tb1tcb(RowWindowOffset, SparseAToXindex, TCblockBitMap, 
                                                 size, Q_half, K_half, V_half, apply_softmax, save_sddmm_result)
      else:
        raise ValueError(f"Invalid algorithm: {args.alg}")
    print(f"f3s_time: {time.item()} ms")
    if args.check_sddmm:
      rel_err = torch.norm(sddmm_result - sddmm_true) / sddmm_true_norm
      f3s_v_true_fp32_sddmm.append(rel_err.item())

    torch.set_printoptions(precision=2, threshold=float('inf'))
    if fusedR is not None:
      print(fusedR.shape)
      print(true.shape)
      # for i in range(1):
      #   print(fusedR[:, :])
      # print("--------------------------------")
      # for i in range(1):
      #   print(true[:, :])

      diff = fusedR - true
      max_diff = torch.max(torch.abs(diff))
      max_idx = torch.argmax(torch.abs(diff))
      row_idx = max_idx // diff.size(1)
      col_idx = max_idx % diff.size(1)
      print(f"Max absolute difference: {max_diff:.6f} at position ({row_idx}, {col_idx}), original value: {true[row_idx, col_idx]}, f3s value: {fusedR[row_idx, col_idx]}")
      rel_err = torch.norm(fusedR - true) / true_norm
      f3s_v_true_fp32_final.append(rel_err.item())

  half_v_float_final_mean = np.mean(np.array(half_v_float_final))
  print(f"final solution pytorch half vs float: {half_v_float_final_mean}")

  if args.check_sddmm:
    half_v_float_sddmm_mean = np.mean(np.array(half_v_float_sddmm))
    print(f"sddmm pytorch half vs single: {half_v_float_sddmm_mean}")
    f3s_v_true_fp32_sddmm = np.mean(np.array(f3s_v_true_fp32_sddmm))
    print(f"f3s sddmm vs pytorch float: {f3s_v_true_fp32_sddmm}")

  if fusedR is not None:
    f3s_v_true_fp32_final = np.mean(np.array(f3s_v_true_fp32_final))
    print(f"f3s final solution vs pytorch float: {f3s_v_true_fp32_final}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--embedding_size", '-emb', type=int, default=128)
  parser.add_argument("--size", '-s', type=int, default=1000)
  parser.add_argument("--density", '-d', type=float, default=0.1)
  parser.add_argument("--skip_softmax", action='store_true')
  parser.add_argument("--check_sddmm", '-c', action='store_true')
  parser.add_argument("--alg", '-a', type=str, default='1tb1rw_scheduled_permuteV', 
                      choices=['1tb1tcb', '1tb1rw', '1tb1rw_scheduled', '1tb1rw_scheduled_permuteV'])
  args = parser.parse_args()
  main(args)


