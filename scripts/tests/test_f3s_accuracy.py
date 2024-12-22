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

# I have a matrix as a pytorch tensor. this fucntion does the following:
# 1. divided it into row blocks of height BLK_M. If the last row block is not full, pad it with zeros.
# 2. Inside each row block, remove all the columns that are all 0s.
# 3. Inside each row block, group the remaining columns into groups of BLK_N. This should result in submatrices of size BLK_M x BLK_N
# 4. Go through these submatrices in each row block. for each submatrices, write their values into R in row-major order.
# 5. return R
def process_matrix(mat, BLK_M, BLK_N):
    M, K = mat.shape
    num_blocks = (M + BLK_M - 1) // BLK_M  # Ceiling division to get the number of row blocks

    submatrices_list = []

    for i in range(num_blocks):
        # Step 1: Divide into row blocks of height BLK_M
        block_start_row = i * BLK_M
        block_end_row = min((i + 1) * BLK_M, M)
        block_rows = mat[block_start_row:block_end_row, :]
        block_height = block_rows.shape[0]

        # Step 2: Remove columns that are all zeros within this block
        non_zero_columns = (block_rows != 0).any(dim=0)
        block_rows = block_rows[:, non_zero_columns]

        # Step 3: Group remaining columns into groups of BLK_N
        num_columns_remaining = block_rows.shape[1]
        num_submatrices_in_block = (num_columns_remaining + BLK_N - 1) // BLK_N  # Ceiling division
        padded_num_columns = num_submatrices_in_block * BLK_N
        pad_columns = padded_num_columns - num_columns_remaining

        # Pad columns with zeros if necessary
        if pad_columns > 0:
            padding = torch.zeros((block_rows.shape[0], pad_columns), dtype=block_rows.dtype, device=block_rows.device)
            block_rows = torch.cat([block_rows, padding], dim=1)

        # Pad rows with zeros if necessary to ensure each block has BLK_M rows
        if block_height < BLK_M:
            pad_rows = BLK_M - block_height
            padding = torch.zeros((pad_rows, block_rows.shape[1]), dtype=block_rows.dtype, device=block_rows.device)
            block_rows = torch.cat([block_rows, padding], dim=0)

        # Reshape and permute to get submatrices of size BLK_M x BLK_N
        block_rows = block_rows.view(BLK_M, num_submatrices_in_block, BLK_N)
        block_submatrices = block_rows.permute(1, 0, 2)  # Shape: (num_submatrices_in_block, BLK_M, BLK_N)

        submatrices_list.append(block_submatrices.flatten())

    # Step 4: Concatenate all submatrices into R
    R = torch.cat(submatrices_list, dim=0)  # Shape: (N, BLK_M, BLK_N)

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
    exp_matrix = torch.exp(shifted_matrix) * mask.to(matrix.dtype)
    row_sum = exp_matrix.sum(dim=1, keepdim=True)
    # Prevent division by zero if a row is entirely masked
    row_sum = torch.where(row_sum == 0, one_val, row_sum)
    softmax_matrix = exp_matrix / row_sum
    return softmax_matrix

def main():
  n_runs = 1
  n_test = 1
  BLK_H = 16
  BLK_W = 8
  n_heads = 1
  feature_size = 320
  size = 1000
  density = 0.2
  # whether to use new parallel strategy where each warp computes a tcb
  warp_tcb = True
  use_1tb1rw = False
  apply_softmax = True
  # for 1tb1rw and 1tbnrw, the number of warps per block
  nWarpPerBlock = 20
  half_v_float_sddmm = []
  half_v_float_softmax = []
  half_v_float_final = []
  f3s_v_true_fp32_sddmm = []
  f3s_v_true_fp32_final = []
  fusedRs = []
  # np.random.seed(26)
  # torch.manual_seed(26)
  for n in range(n_test):
    np.random.seed(n)
    torch.manual_seed(n)

    print(f"----------{n}-----------")
 
    A_csr_h = sp.sparse.random(size, size, density=density, format='csr', data_rvs=np.random.rand)
    # A_csr_h = (A_csr_h + A_csr_h.T) / 2
    padded_size = ((size + BLK_H - 1) // BLK_H) * BLK_H
    A_csr_h_padded = pad_csr_matrix(A_csr_h, padded_size, padded_size)

    # Round and convert all non-zero entries to 1
    A_csr_h.data = np.ceil(A_csr_h.data, dtype=np.float32)
    A_dense = torch.tensor(A_csr_h.todense()).cuda()
    A_dense_half = A_dense.to(torch.float16)

    # generate the dense feature matrix
    Q = torch.rand(size, feature_size, dtype=torch.float32, device='cuda')
    K = torch.rand(size, feature_size, dtype=torch.float32, device='cuda')
    V = torch.rand(size, feature_size, dtype=torch.float32, device='cuda')
    # pad the feature matrix to have a multiple of 16 columns
    # if feature_size % BLK_H != 0:
    col_padding_len = 0 if feature_size % BLK_H == 0 else BLK_H - feature_size % BLK_H
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
    torch.set_printoptions(precision=3)
    
    sddmm_half_og_form = (Q_half @ K_half.T) * A_dense_half
    sddmm_og_form = (Q @ K.T) * A_dense
    sddmm_true_half = process_matrix(sddmm_half_og_form, BLK_H, BLK_W)
    sddmm_true = process_matrix(sddmm_og_form, BLK_H, BLK_W)
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

    save_sddmm_result = True
    num_row_windows = (size + BLK_H - 1) // BLK_H
    edgeToColumn = torch.zeros(A_csr_h.nnz, dtype=torch.int)
    edgeToRow = torch.zeros(A_csr_h.nnz, dtype=torch.int)
    blockPartition = torch.zeros(num_row_windows, dtype=torch.int)
    blockPartition_cuda  = blockPartition.cuda()
    edgeToColumn_cuda = edgeToColumn.cuda()
    edgeToRow_cuda  = edgeToRow.cuda()
    indptr = torch.IntTensor(A_csr_h.indices).cuda()
    indices = torch.IntTensor(A_csr_h.indptr).cuda()
    RowWindowOffset, TCblockRowid,\
    TCblocktileId, TCblockoffset, SparseAToXindex,\
    TBBoundaries, TCblockBitMap, block_count = TCFMM.preprocess_gpu(indptr, indices, size, BLK_H, BLK_W, blockPartition_cuda, edgeToColumn_cuda, edgeToRow_cuda)
    start_time = time.time()
    for i in range(n_runs):
      if warp_tcb:
        sddmm_result = TCFMM.f3S_sddmm(RowWindowOffset, TBBoundaries, TCblockRowid, SparseAToXindex, TCblockBitMap, size, Q_half, K_half, nWarpPerBlock, use_1tb1rw)[0]
        fusedR = None
      else:
        fusedR, sddmm_result = TCFMM.f3S_forward(RowWindowOffset, SparseAToXindex, TCblockBitMap, size, Q_half, K_half, V_half, apply_softmax, save_sddmm_result)
    f3s_time = (time.time() - start_time)/n_runs
    print(f"f3s_time: {f3s_time}")
    rel_err = torch.norm(sddmm_result - sddmm_true) / sddmm_true_norm
    f3s_v_true_fp32_sddmm.append(rel_err.item())
    if fusedR is not None:
      rel_err = torch.norm(fusedR - true) / true_norm
      f3s_v_true_fp32_final.append(rel_err.item())

  half_v_float_sddmm_mean = np.mean(np.array(half_v_float_sddmm))
  print(f"sddmm pytorch half vs single: {half_v_float_sddmm_mean}")

  half_v_float_final_mean = np.mean(np.array(half_v_float_final))
  print(f"final solution pytorch half vs float: {half_v_float_final_mean}")

  f3s_v_true_fp32_sddmm = np.mean(np.array(f3s_v_true_fp32_sddmm))
  print(f"f3s sddmm vs pytorch float: {f3s_v_true_fp32_sddmm}")

  if fusedR is not None:
    f3s_v_true_fp32_final = np.mean(np.array(f3s_v_true_fp32_final))
    print(f"f3s final solution vs pytorch float: {f3s_v_true_fp32_final}")


if __name__ == "__main__":
  main()

