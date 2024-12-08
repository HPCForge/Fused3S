#include "config.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <mma.h>
#include <sstream>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <torch/extension.h>
#include <vector>
#include "ptx.h"

using namespace nvcuda;

union half2_uint32 {
    half2 h2;
    uint32_t u32;
};

// Assume each warp has a 8x8 fp16 matrix in row-major order distributed among threads
// Each thread starts with 2 consecutive fp16 values as a half2 (val)
// This function redistributes elements among threads 
// so that val becomes 2 consecutive fp16 values in column-major order
__device__ void shfl_transpose_warp(half2 &val, int laneid){
  int col = laneid/4;
  int row = (laneid%4)*2;
  half2 temp[2];
  temp[0] = __shfl_sync(0xffffffff, val, row*4 + col/2);
  temp[1] = __shfl_sync(0xffffffff, val, (row+1)*4 + col/2);
  if((laneid/4)%2 == 0){
    val.x = temp[0].x;
    val.y = temp[1].x;
  }
  else{
    val.x = temp[0].y;
    val.y = temp[1].y;
  }
}

// assuming each tcblock can be divided into 8x8 (64 bits) sub-blocks
// local_id is the id of the element in tcblock
__device__ void update_bitmap(uint64_t* bitmap, 
                              int tcblock_id, int n_sub_blocks_per_tcblock, int local_id) {
  unsigned long long int *ull_bitmap = reinterpret_cast<unsigned long long int*>(bitmap);
  int sub_block_id = local_id / 64;
  uint64_t mask = 1ULL << (63 - local_id % 64);
  atomicOr(&ull_bitmap[tcblock_id * n_sub_blocks_per_tcblock + sub_block_id], mask);
}

//////////////////////////////////////////////////////////////////////
/// Preprocessing
//////////////////////////////////////////////////////////////////////
__global__ void roundup_to_multiple_of_eight(int *input, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    int rounded_value = ((input[tid] + 7) / 8) * 8;
    input[tid] = rounded_value;
  }
}

__global__ void get_padding_tileid_kernel(int *ori_offset, uint8_t *ori_tileid,
                                          int *padded_offset,
                                          uint8_t *padded_tileid, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    int s = ori_offset[tid];
    int e = ori_offset[tid + 1];
    int s1 = padded_offset[tid];
    for (int i = 0; i < e - s; i++) {
      padded_tileid[s1 + i] = ori_tileid[s + i];
    }
  }
}

__global__ void fill_edgeToRow(int *edgeToRow, int *nodePointer,
                               int num_nodes) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int nid = tid / 32;
  int laneid = tid % 32;
  // check a valid node range.
  if (nid < num_nodes) {
#pragma unroll
    for (int eid = nodePointer[nid] + laneid; eid < nodePointer[nid + 1];
         eid += 32) {
      edgeToRow[eid] = nid;
    }
  }
}
/*Generate segment*/
__global__ void fill_segment(int *nodePointer, int *seg_out, int blockSize_h,
                             int blockSize_w, int num_nodes) {
  int tid = threadIdx.x;
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = nodePointer[winId * blockSize_h];
  unsigned block_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  unsigned num_window_edges = block_end - block_start;
  const unsigned threadPerBlock = blockDim.x * blockDim.y;
  for (unsigned idx = tid; idx < num_window_edges; idx += threadPerBlock) {
    seg_out[block_start + idx] = winId;
  }
}
void fill_segment_cuda(int *nodePointer, int *seg_out, int blockSize_h,
                       int blockSize_w, int num_nodes) {
  int block_size = 512;
  int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
  fill_segment<<<window_count, block_size>>>(nodePointer, seg_out, blockSize_h,
                                             blockSize_w, num_nodes);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

/*Generate TCblock_rowid*/
__global__ void generate_tcblock_rowid(int *rowwindow_offset,
                                       int *tcblock_rowid,
                                       int num_row_windows) {
  int tid = threadIdx.x;
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = rowwindow_offset[winId];
  unsigned block_end = rowwindow_offset[min(winId + 1, num_row_windows)];
  unsigned num_blocks = block_end - block_start;
  const unsigned threadPerBlock = blockDim.x * blockDim.y;
  for (unsigned idx = tid; idx < num_blocks; idx += threadPerBlock) {
    tcblock_rowid[block_start + idx] = winId;
  }
}
void generate_tcblock_rowid_cuda(int *rowwindow_offset, int *tcblock_rowid,
                                 int num_row_windows) {
  int block_size = 512;
  int window_count = num_row_windows;
  generate_tcblock_rowid<<<window_count, block_size>>>(
      rowwindow_offset, tcblock_rowid, num_row_windows);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

/* Generate edge2column*/
__device__ __forceinline__ int binarysearch(int *arr, int size, int target) {
  int left = 0;
  int right = size - 1;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (arr[mid] == target) {
      while (mid > 0 && arr[mid - 1] == target) {
        mid--;
      }
      return mid;
    } else if (arr[mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return -1;
}
__device__ __forceinline__ void inplace_deduplication(int *array, int length,
                                                      int *loc) {
  int cur = 1;
  while (cur < length) {
    if (array[cur] != array[cur - 1]) {
      (*loc)++;
      array[(*loc)] = array[cur];
    }
    cur++;
  }
}
__global__ void generate_edgetocolumn(int *nodePointer, int *edgelist,
                                      int *edgelist_sort, int *edgetocol,
                                      int *blockpartition, int *blocknum,
                                      int blockSize_h, int blockSize_w,
                                      int num_nodes) {
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = nodePointer[winId * blockSize_h];
  unsigned block_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  unsigned num_window_edges = block_end - block_start;
  if (num_window_edges == 0)
    return;
  int *start = edgelist_sort + block_start;
  int size = 0;
  inplace_deduplication(start, num_window_edges, &size);
  int num = (size + blockSize_w) / blockSize_w;
  atomicAdd(blocknum, num);
  blockpartition[winId] = num;
  for (unsigned idx = block_start; idx < block_end; idx += 1) {
    int index = binarysearch(start, size + 1, edgelist[idx]);
    edgetocol[idx] = index;
  }
}
void generate_edgetocolumn_cuda(int *nodePointer, int *edgelist,
                                int *edgelist_sort, int *edgetocol,
                                int *blockpartition, int *blocknum,
                                int blockSize_h, int blockSize_w,
                                int num_nodes) {
  int block_size = 1;
  int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
  generate_edgetocolumn<<<window_count, block_size>>>(
      nodePointer, edgelist, edgelist_sort, edgetocol, blockpartition, blocknum,
      blockSize_h, blockSize_w, num_nodes);
  // generate_edgetocolumn_v1<<< window_count, block_size >>> (nodePointer,
  // edgelist, edgelist_sort, edgetocol, blockpartition, blocknum, blockSize_h,
  // blockSize_w, num_nodes);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

/*Generate TC offset, tileid and AtoB*/
__global__ void generate_tcoffset_id_atob(
    int *nodePointer, int *rowwindow_offset, int *edgeToColumn, int *edgeToRow,
    int *edgeList, int *tcblock_offset, uint8_t *tcblocktile_id,
    int *sparseatob, uint64_t *tcblock_bit_map, 
    int max_block, int num_nodes, int blockSize_h,
    int blockSize_w, int num_row_windows) {
  extern __shared__ int pos_ptr[];
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = rowwindow_offset[winId];
  unsigned block_end = rowwindow_offset[min(winId + 1, num_row_windows)];
  unsigned num_blocks = block_end - block_start;
  if (num_blocks == 0) {
    return;
  }
  int *tcblock_offset_ptr = pos_ptr + num_blocks;
  int *tcblock_offset_global_ptr = tcblock_offset + block_start;
  int *tcblock_nnz_ptr = pos_ptr + num_blocks + 1;
  unsigned element_start = nodePointer[winId * blockSize_h];
  unsigned element_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  unsigned num_window_edges = element_end - element_start;
  if (num_window_edges == 0) {
    return;
  }
  for (int i = 0; i < 2 * num_blocks + 1; i++) {
    pos_ptr[i] = 0;
  }
  for (unsigned e_index = element_start; e_index < element_end; e_index++) {
    unsigned col = edgeToColumn[e_index]; // new col
    tcblock_nnz_ptr[col / blockSize_w]++;
  }
  for (int i = 0; i < num_blocks; i++) {
    tcblock_offset_global_ptr[i] = tcblock_nnz_ptr[i];
  }
  auto tileid = tcblocktile_id + element_start;
  auto sparse_AToB = sparseatob + block_start * blockSize_w;
  for (int i = 0; i < num_blocks; i++) {
    tcblock_nnz_ptr[i] += tcblock_nnz_ptr[i - 1];
  }
  int n_sub_blocks_per_tcblock = blockSize_w * blockSize_h / 64;
  for (unsigned e_index = element_start; e_index < element_end; e_index++) {
    unsigned col = edgeToColumn[e_index]; // new col
    unsigned tcblock_id = col / blockSize_w;
    unsigned row_local = edgeToRow[e_index] % blockSize_h;
    unsigned col_local = col % blockSize_w;
    tileid[tcblock_offset_ptr[tcblock_id] + pos_ptr[tcblock_id]] =
        (uint8_t)(row_local * blockSize_w + col_local);
    update_bitmap(tcblock_bit_map, block_start + tcblock_id, n_sub_blocks_per_tcblock, int(row_local * blockSize_w + col_local));
    sparse_AToB[tcblock_id * blockSize_w + col_local] = edgeList[e_index];
    pos_ptr[tcblock_id]++;
  }
}
void generate_tcoffset_id_atob_cuda(int *nodePointer, int *rowwindow_offset,
                                    int *edgeToColumn, int *edgeToRow,
                                    int *edgeList, int *tcblock_offset,
                                    uint8_t *tcblock_tileid, int *sparseatob,
                                    uint64_t *tcblock_bit_map,
                                    int max_block, int num_nodes,
                                    int blockSize_h, int blockSize_w,
                                    int num_row_windows) {
  int block_size = 1;
  int window_count = num_row_windows;
  const int dynamic_shared_size = (2 * max_block + 1) * sizeof(int);
  std::cout << "dynamic_shared_size: " << dynamic_shared_size << std::endl;
  if (dynamic_shared_size > 98304) {
    int maxbytes = 131072; // 96 KB
    cudaFuncSetAttribute(generate_tcoffset_id_atob,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  } else if (dynamic_shared_size > 65536) {
    int maxbytes = 98304; // 96 KB
    cudaFuncSetAttribute(generate_tcoffset_id_atob,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  } else if (dynamic_shared_size > 32768) {
    int maxbytes = 65536; // 128 KB
    cudaFuncSetAttribute(generate_tcoffset_id_atob,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  }
  generate_tcoffset_id_atob<<<window_count, block_size, dynamic_shared_size>>>(
      nodePointer, rowwindow_offset, edgeToColumn, edgeToRow, edgeList,
      tcblock_offset, tcblock_tileid, sparseatob, tcblock_bit_map, max_block, num_nodes,
      blockSize_h, blockSize_w, num_row_windows);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}
void padding_up_8(int *input, int size) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  roundup_to_multiple_of_eight<<<blocksPerGrid, threadsPerBlock>>>(input, size);
}
void get_padding_tileid(int *ori_offset, uint8_t *ori_tileid,
                        int *padded_offset, uint8_t *padded_tileid, int size) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  get_padding_tileid_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      ori_offset, ori_tileid, padded_offset, padded_tileid, size);
}
/*main function*/
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
seg_sort_dequ(int *seg, int *edgeLists, int *nodepointer, int *edgetocol,
              int *edgetorow, int *blockpartition, int *block_num,
              int *rowwindow_offset, int blockSize_h, int blockSize_w,
              int num_nodes, int num_edges, int rowwindow_num) {
  thrust::device_ptr<int> Seg = thrust::device_pointer_cast(seg);
  thrust::device_vector<int> deviceSeg(Seg, Seg + num_edges);
  thrust::device_ptr<int> EL = thrust::device_pointer_cast(edgeLists);
  thrust::device_vector<int> deviceEL(EL, EL + num_edges);
  auto begin = thrust::make_zip_iterator(
      thrust::make_tuple(deviceSeg.begin(), deviceEL.begin()));
  auto end = thrust::make_zip_iterator(
      thrust::make_tuple(deviceSeg.end(), deviceEL.end()));
  thrust::sort(thrust::device, begin, end);
  generate_edgetocolumn_cuda(
      nodepointer, edgeLists, thrust::raw_pointer_cast(&deviceEL[0]), edgetocol,
      blockpartition, block_num, blockSize_h, blockSize_w, num_nodes);
  thrust::device_ptr<int> blockpartition_ptr =
      thrust::device_pointer_cast(blockpartition);
  thrust::device_ptr<int> rowwindow_offset_ptr =
      thrust::device_pointer_cast(rowwindow_offset + 1);
  thrust::device_vector<int> blockpartition_vector(
      blockpartition_ptr, blockpartition_ptr + rowwindow_num);
  thrust::inclusive_scan(blockpartition_vector.begin(),
                         blockpartition_vector.end(), rowwindow_offset_ptr);
  auto options_gpu =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  auto options_gpu_unit8 =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
  auto options_gpu_uint64 =
      torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCUDA);
  thrust::device_ptr<int> bnum_ptr = thrust::device_pointer_cast(block_num);
  thrust::host_vector<int> bnum_vector(bnum_ptr, bnum_ptr + 1);
  int block_counter = bnum_vector[0];
  auto tcblock_rowid_tensor = torch::zeros({block_counter}, options_gpu);
  auto tcblock_rowid = tcblock_rowid_tensor.data_ptr<int>();
  generate_tcblock_rowid_cuda(rowwindow_offset, tcblock_rowid, rowwindow_num);
  auto max_element =
      thrust::max_element(thrust::device, blockpartition_vector.begin(),
                          blockpartition_vector.end());
  int max_blocks = *max_element;
  auto tcblocktile_id_tensor = torch::zeros({num_edges}, options_gpu_unit8);
  auto tcblock_offset_tensor = torch::zeros({block_counter + 1}, options_gpu);
  auto sparse_AToX_index_tensor =
      torch::zeros({block_counter * blockSize_w}, options_gpu);
  int bit_map_size = BLK_M * BLK_N;
  assert(bit_map_size % 64 == 0);
  int bit_map_int64_size = bit_map_size / 64;
  auto tcblock_bit_map_tensor = torch::zeros({block_counter*bit_map_int64_size}, options_gpu_uint64);
  auto tcblock_bit_map = tcblock_bit_map_tensor.data_ptr<uint64_t>();
  auto tcblock_offset = tcblock_offset_tensor.data_ptr<int>();
  auto sparse_AToX_index = sparse_AToX_index_tensor.data_ptr<int>();
  auto tcblocktile_id = tcblocktile_id_tensor.data_ptr<uint8_t>();
  generate_tcoffset_id_atob_cuda(
      nodepointer, rowwindow_offset, edgetocol, edgetorow, edgeLists,
      tcblock_offset + 1, tcblocktile_id, sparse_AToX_index, tcblock_bit_map, max_blocks,
      num_nodes, blockSize_h, blockSize_w, rowwindow_num);
  thrust::device_ptr<int> tcblock_offset_ptr =
      thrust::device_pointer_cast(tcblock_offset);
  thrust::inclusive_scan(tcblock_offset_ptr,
                         tcblock_offset_ptr + block_counter + 1,
                         tcblock_offset_ptr);
  return std::make_tuple(tcblock_offset_tensor, tcblock_rowid_tensor,
                         tcblocktile_id_tensor, sparse_AToX_index_tensor,
                         tcblock_bit_map_tensor, block_counter);
}
void fill_edgeToRow_cuda(int *edgeToRow, int *nodePointer, int num_nodes) {
  int wrap_size = 32;
  int block_size = 1024;
  int grid_size = (num_nodes * wrap_size + block_size - 1) / block_size;
  fill_edgeToRow<<<grid_size, block_size>>>(edgeToRow, nodePointer, num_nodes);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}



__global__ void TC_fusedMM_cuda_kernel(
	const int *__restrict__ TCblock_rowid, 		 // offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id,  // id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset,      // colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx,     // colid of each TC block nonzero element.
	const int numNodes, const int numEdges,
	const int embedding_dim,    // embedding dimension.
	float *__restrict__ in_mat, // input feature matrix.
	float *output,              // aggreAGNNed output feature matrix.
	torch::Half *edgeAttention, // result of SDDMM.
	bool save_edge_attention
);
__global__ void TC_fusedMM_fp32_inter_cuda_kernel(
	const int *__restrict__ TCblock_rowid, 		 // offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id,  // id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset,      // colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx,     // colid of each TC block nonzero element.
	const int numNodes, const int numEdges,
	const int embedding_dim,    // embedding dimension.
	torch::Half *__restrict__ in_mat, // input feature matrix.
	float *output,              // aggreAGNNed output feature matrix.
	float *edgeAttention, // result of SDDMM.
	bool save_edge_attention
);
#if BLK_M == 8 && BLK_N == 32 && BLK_K == 16
__global__ void TC_fusedMM_fp32_inter_m8n32k16_cuda_kernel(
		const int *__restrict__ TCblock_rowid, 		 // offset of each row window.
		const uint8_t *__restrict__ TCblocktile_id,  // id of each TC block nonzero element.
		const int *__restrict__ TCblock_offset,      // colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx,     // colid of each TC block nonzero element.
		const int numNodes, const int numEdges,
		const int embedding_dim,    // embedding dimension.
		torch::Half *__restrict__ in_mat, // input feature matrix.
		float *output,              // aggreAGNNed output feature matrix.
		float *edgeAttention, // result of SDDMM.
		bool save_edge_attention);
#endif

__global__ void f3s_m16n8k16_cuda_kernel(
		const int *__restrict__ TCblock_rowid, 		 // offset of each row window.
		const int *__restrict__ sparse_AToX_idx,     // colid of each TC block nonzero element.
    const uint64_t *__restrict__ TCblock_bit_map,
    const int numNodes,
    const int embedding_dim,
		torch::Half *__restrict__ Q, 
    torch::Half *__restrict__ K, 
    torch::Half *__restrict__ V,
		float *output,              // output feature matrix.
		float *edgeAttention // result of SDDMM
);

std::vector<torch::Tensor> f3S_forward_cuda(
    torch::Tensor TCblock_rowid,
    torch::Tensor sparse_AToX_idx, 
    torch::Tensor TCblock_bit_map,
    int num_nodes, 
    int embedding_dim,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, 
    bool save_sddmm_result){
  int nBlockEmbeddingDim = (embedding_dim + BLK_N - 1) / BLK_N;
  int nWarpPerBlock = (nBlockEmbeddingDim + TCBLOCK_PER_WARP - 1) / TCBLOCK_PER_WARP;
  const int nRowWindow = TCblock_rowid.size(0) - 1;
  int paddedLength = nRowWindow * BLK_M;
  auto output = torch::zeros({paddedLength, embedding_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  dim3 grid(nRowWindow, 1, 1);
  dim3 block(WARP_SIZE, nWarpPerBlock, 1);
  int nTCBlock = sparse_AToX_idx.size(0)/BLK_N;
	torch::Tensor sddmm_result = torch::zeros({nTCBlock*BLK_M*BLK_N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));;
  int dynamic_shared_size = nWarpPerBlock * BLK_M * BLK_N * 2 * sizeof(float);
  float* sddmm_result_ptr = save_sddmm_result ? sddmm_result.data_ptr<float>() : nullptr;
  #if BLK_M == 16 && BLK_N == 8 && BLK_K == 16
  f3s_m16n8k16_cuda_kernel<<<grid, block, dynamic_shared_size>>>(
    TCblock_rowid.data_ptr<int>(), 
    sparse_AToX_idx.data_ptr<int>(),
    TCblock_bit_map.data_ptr<uint64_t>(),
    num_nodes, embedding_dim,
    Q.data_ptr<torch::Half>(), 
    K.data_ptr<torch::Half>(), 
    V.data_ptr<torch::Half>(),
    output.data_ptr<float>(),
    sddmm_result_ptr);
  #else
  printf("only m16n8k16 is supported\n");
  #endif

  // check for error
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
  // remove padding
  output = output.index(
      {torch::indexing::Slice(0, num_nodes), torch::indexing::Slice()});
  return {output, sddmm_result};
}

std::vector<torch::Tensor> fusedMM_forward_cuda(
	torch::Tensor Rowwindow_offset,
	torch::Tensor TCblocktile_id,
	torch::Tensor TCblock_offset,
	torch::Tensor sparse_AToX_idx,
	int num_nodes, int num_edges,
	int embedding_dim,  // embedding dimension.
	torch::Tensor input, // input feature matrix.
	bool save_edge_attention,
	bool use_f32_edge_attention,
	// default to m16n16k16
	bool use_m8n32k16 ) {
  // warps per block
  const int num_row_windows = Rowwindow_offset.size(0) - 1;
	int row_window_height;
	int nWarpPerBlock;
	if(use_m8n32k16){
		row_window_height = BLK_M;
		// Assuming embedding_dim is a multiple of 2*BLK_K
		nWarpPerBlock = (embedding_dim + BLK_K - 1) / BLK_K / TCBLOCK_PER_WARP_FMM;
	} else {
		row_window_height = BLK_H;
		nWarpPerBlock = (embedding_dim + row_window_height - 1) / row_window_height;
	}
	int paddedLength = num_row_windows * row_window_height;
  auto output = torch::zeros({paddedLength, embedding_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  dim3 grid(num_row_windows, 1, 1);
  dim3 block(WARP_SIZE, nWarpPerBlock, 1);
	torch::Tensor edgeAttention;
	if(use_f32_edge_attention) {
		edgeAttention = torch::zeros_like(TCblocktile_id).to(torch::kFloat32);
		if(use_m8n32k16){
      #if BLK_M == 8 && BLK_N == 32 && BLK_K == 16
			int dynamic_shared_size = nWarpPerBlock * (BLK_M * BLK_N * sizeof(float) + BLK_N * BLK_N * sizeof(half));
			TC_fusedMM_fp32_inter_m8n32k16_cuda_kernel<<<grid, block, dynamic_shared_size>>>(
			Rowwindow_offset.data_ptr<int>(), 
			TCblocktile_id.data_ptr<uint8_t>(),
			TCblock_offset.data_ptr<int>(), 
			sparse_AToX_idx.data_ptr<int>(),
			num_nodes, num_edges, embedding_dim,
			input.data_ptr<torch::Half>(), 
			output.data_ptr<float>(),
			edgeAttention.data_ptr<float>(), 
			save_edge_attention);
      #else
      printf("m8n32k16 is not supported\n");
      #endif
		}
		else{
			int dynamic_shared_size = nWarpPerBlock * BLK_H * BLK_H * (sizeof(half) + sizeof(float));
			TC_fusedMM_fp32_inter_cuda_kernel<<<grid, block, dynamic_shared_size>>>(
				Rowwindow_offset.data_ptr<int>(), 
				TCblocktile_id.data_ptr<uint8_t>(),
				TCblock_offset.data_ptr<int>(), 
				sparse_AToX_idx.data_ptr<int>(),
				num_nodes, num_edges, embedding_dim,
				input.data_ptr<torch::Half>(), 
				output.data_ptr<float>(),
				edgeAttention.data_ptr<float>(), 
				save_edge_attention);
		}
	}
	else{
		edgeAttention = torch::zeros_like(TCblocktile_id).to(torch::kHalf);
		const int dynamic_shared_size = 3 * nWarpPerBlock * BLK_H * BLK_H * sizeof(half);
		TC_fusedMM_cuda_kernel<<<grid, block, dynamic_shared_size>>>(
			Rowwindow_offset.data_ptr<int>(), 
			TCblocktile_id.data_ptr<uint8_t>(),
			TCblock_offset.data_ptr<int>(), 
			sparse_AToX_idx.data_ptr<int>(),
			num_nodes, num_edges, embedding_dim,
			input.data_ptr<float>(), 
			output.data_ptr<float>(),
			edgeAttention.data_ptr<torch::Half>(), 
			save_edge_attention);
	}
  // check for error
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
  // remove padding
  output = output.index(
      {torch::indexing::Slice(0, num_nodes), torch::indexing::Slice()});
  return {output, edgeAttention};
}

//////////////////////
/// fusedMM
/// should be launched with (embedding_dim + 16 - 1) / 16 warps of 32 threads
/// note here we are assuming only 1 attention head
//////////////////////
__global__ void TC_fusedMM_cuda_kernel(
		const int *__restrict__ TCblock_rowid, 		 // offset of each row window.
		const uint8_t *__restrict__ TCblocktile_id,  // id of each TC block nonzero element.
		const int *__restrict__ TCblock_offset,      // colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx,     // colid of each TC block nonzero element.
    const int numNodes, const int numEdges,
    const int embedding_dim,    // embedding dimension.
    float *__restrict__ in_mat, // input feature matrix.
    float *output,              // aggreAGNNed output feature matrix.
    torch::Half *edgeAttention, // result of SDDMM.
    bool save_edge_attention) {
  int bid = blockIdx.x;     // block_index == row_window_index
  int wid = threadIdx.y;    // warp_index handling multi-dimension > 16.
  int laneid = threadIdx.x; // lanid of each warp.
  int tid = wid * blockDim.x + laneid; // threadid of each block.

  int warpPerBlock = blockDim.y;
  int threadPerBlock = blockDim.x * warpPerBlock;

  // starting node_id of current row_window.
  int nid_start = bid * BLK_H; 
  // ending node_id of the current row_window.
  int nid_end = min((bid + 1) * BLK_H, numNodes); 
  assert(nid_start < nid_end);

  int warp_offset = wid * BLK_H * BLK_H;
  __shared__ half edge_attention_block[BLK_H * BLK_H]; // Result of SDDMM

  // each warp uses 3x16x16 shared memory
  extern __shared__ half dynamic_shared[]; // 3 x blockDim.y x 16 x 16.
  half *sparse_A_val = dynamic_shared;     // result of XX^T for all warps
  half *dense_X_lst =
      dynamic_shared + warpPerBlock * BLK_H * BLK_H; // X for all warps
  half *dense_Y_lst =
      dynamic_shared + 2 * warpPerBlock * BLK_H * BLK_H; // X^T for all warps

  wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_H, half, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_H, half, wmma::col_major>
      b_frag;
  wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_H, half, wmma::row_major>
      b_t_frag;
  wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_H, half> acc_frag;
  wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_H, float> spmm_acc_frag;
  wmma::fill_fragment(spmm_acc_frag, 0.0f);

  // Processing TC_blocks along the column dimension of Sparse A.
  ///////////////////////////////////////////////////////////////
  // Initialize dense_X by row-major store,
  // Threads of a warp for fetching a dense_X.
  // TODO: this is not coalesced. Make every 2 warps fetch their dense_X together to coalesce.
#pragma unroll
  for (int i = laneid; i < BLK_H * BLK_H; i += blockDim.x) {
    int dense_rowIdx = i / BLK_H;
    int dense_dimIdx = i % BLK_H;
    int source_idx =
        (nid_start + dense_rowIdx) * embedding_dim + wid * BLK_H + dense_dimIdx;
    if (source_idx >= numNodes * embedding_dim)
      dense_X_lst[i + warp_offset] = __float2half(0.0f);
    else
      dense_X_lst[i + warp_offset] = __float2half(in_mat[source_idx]);
  }

  /////////////////////////////////
  // main loop
  /////////////////////////////////
  for (int tcb_id = TCblock_rowid[bid]; tcb_id < TCblock_rowid[bid+1]; tcb_id++) {
	// TODO: is this necessary? Feels like the one on line 5472 is enough
    __syncthreads();

#pragma unroll
    for (int idx = tid; idx < BLK_H * BLK_H; idx += threadPerBlock) {
      edge_attention_block[idx] = __float2half(0.0f);
    }

// Initialize dense_Y by column-major store,
// Threads of a warp for fetching a dense_Y.
// TODO: this is also not coalesced
#pragma unroll
    for (int i = laneid; i < BLK_H * BLK_H; i += blockDim.x) {
	  // TC block col ind to dense X row ind
      int dense_rowIdx = sparse_AToX_idx[tcb_id * BLK_H + i / BLK_H]; 
      // embedding_dim index of the dense tile.
      int dense_dimIdx = i % BLK_H;    
      int source_idx =
          dense_rowIdx * embedding_dim + wid * BLK_H + dense_dimIdx;
      if (source_idx >= numNodes * embedding_dim) {
        dense_Y_lst[i + warp_offset] = __float2half(0.0f);
      } else {
        dense_Y_lst[i + warp_offset] = __float2half(in_mat[source_idx]);
      }
    }

    wmma::load_matrix_sync(a_frag, dense_X_lst + warp_offset, BLK_H);
    wmma::load_matrix_sync(b_frag, dense_Y_lst + warp_offset, BLK_H);
    // clear acc_frag
    wmma::fill_fragment(acc_frag, __float2half(0.0f));
    // Perform the matrix multiplication on Tensor Core
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    // Each warp store the result into the right slice of the intermediate
    // tensor
    wmma::store_matrix_sync(sparse_A_val + warp_offset, acc_frag, BLK_H,
                            wmma::mem_row_major);

    __syncthreads();

		// Starting and ending ind of nnz of the current TCblock in TCblocktile_id
		int eIdx_start = TCblock_offset[tcb_id];
		int nnz_in_block = TCblock_offset[tcb_id + 1] - eIdx_start;
			// Add up results from all warps using cuda cores
		// NOTE: if threadPerBlock > nnz_in_block, we only need 1 iteration. 
		// But if threadPerBlock < nnz_in_block, we need multiple iterations, which could slow things down.
		for(int i = tid; i < nnz_in_block; i += threadPerBlock) {
			int eid = eIdx_start + i;
			int block_id = TCblocktile_id[eid];
			for (int j = 0; j < warpPerBlock; j++) {
				// TODO: this access pattern doesn't seem very efficient
				edge_attention_block[block_id] =
					__hadd(edge_attention_block[block_id],
							sparse_A_val[block_id + j * BLK_H * BLK_H]);
			}
			// Save the edge attention
			// TODO: maybe we can coalesce this write
			if(save_edge_attention) {
				edgeAttention[eid] = edge_attention_block[block_id];
			}
		}
    // necessary to ensure edge_attention_block is correct
    __syncthreads();

    /////////
    // SpMM
    /////////
    // load the result of SDDMM
    wmma::load_matrix_sync(a_frag, edge_attention_block, BLK_H);
    // load feature matrix block
    wmma::load_matrix_sync(b_t_frag, dense_Y_lst + warp_offset, BLK_H);

    // spmm_acc_frag might be moved in and out of registers to local(global)
    // memory every iteration, which can be bad (Ampere have 255 registers (32
    // bits each) per thread) so we have to monitor for this, which I'm not sure
    // how alternative is to compute and store the SDDMM results for all tc
    // blocks first this will need (blockDim.y * 16 * 16 * 4) bytes
    wmma::mma_sync(spmm_acc_frag, a_frag, b_t_frag, spmm_acc_frag);
  }
  wmma::store_matrix_sync(output + bid * BLK_H * embedding_dim + wid * BLK_H,
                          spmm_acc_frag, embedding_dim, wmma::mem_row_major);
}


//////////////////////
/// Same as fusedMM
/// Except the partial result for SDDMM is stored in fp32 instead of half
/// note here we are assuming only 1 attention head
//////////////////////
__global__ void TC_fusedMM_fp32_inter_cuda_kernel(
		const int *__restrict__ TCblock_rowid, 		 // offset of each row window.
		const uint8_t *__restrict__ TCblocktile_id,  // id of each TC block nonzero element.
		const int *__restrict__ TCblock_offset,      // colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx,     // colid of each TC block nonzero element.
		const int numNodes, const int numEdges,
		const int embedding_dim,    // embedding dimension.
		torch::Half *__restrict__ in_mat, // input feature matrix.
		float *output,              // aggreAGNNed output feature matrix.
		float *edgeAttention, // result of SDDMM.
		bool save_edge_attention) {
  int bid = blockIdx.x;     // block_index == row_window_index
  int wid = threadIdx.y;    // warp_index handling multi-dimension > 16.
  int laneid = threadIdx.x; // lanid of each warp.
  int tid = wid * blockDim.x + laneid; // threadid of each block.

  int warpPerBlock = blockDim.y;
  int threadPerBlock = blockDim.x * warpPerBlock;

  // starting node_id of current row_window.
  int nid_start = bid * BLK_H; 
  // ending node_id of the current row_window.
  int nid_end = min((bid + 1) * BLK_H, numNodes); 
  assert(nid_start < nid_end);

  int warp_offset = wid * BLK_H * BLK_H;
  __shared__ half edge_attention_block_half[BLK_H * BLK_H]; // Result of SDDMM
	__shared__ float edge_attention_block_single[BLK_H * BLK_H]; // Result of SDDMM in single

  // each warp uses 2x16x16 shared memory
  extern __shared__ char dynamic_shared_mixed[]; // 2 x blockDim.y x 16 x 16.
  float* sparse_A_val = (float*) &dynamic_shared_mixed[0];     // result of XX^T for all warps
  half* dense_X_lst =
      (half*) &dynamic_shared_mixed[warpPerBlock * BLK_H * BLK_H * sizeof(float)]; // X for all warps

  wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_H, half, wmma::row_major>
      a_frag;
	wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_H, half, wmma::row_major>
      att_frag;
  wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_H, half, wmma::col_major>
      b_frag;
  wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_H, half, wmma::row_major>
      b_t_frag;
  wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_H, float> acc_frag;
  wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_H, float> spmm_acc_frag;
  wmma::fill_fragment(spmm_acc_frag, 0.0f);

  // Processing TC_blocks along the column dimension of Sparse A.
  ///////////////////////////////////////////////////////////////
  // Initialize dense_X by row-major store,
  // Threads of a warp for fetching a dense_X.
  // TODO: this is not coalesced. Make every 2 warps fetch their dense_X together to coalesce.
#pragma unroll
  for (int i = laneid; i < BLK_H * BLK_H; i += blockDim.x) {
    int dense_rowIdx = i / BLK_H;
    int dense_dimIdx = i % BLK_H;
    int source_idx =
        (nid_start + dense_rowIdx) * embedding_dim + wid * BLK_H + dense_dimIdx;
    if (source_idx >= numNodes * embedding_dim)
	  // TODO: is this conversion necessary?
      dense_X_lst[i + warp_offset] = __float2half(0.0f);
    else
      dense_X_lst[i + warp_offset] = in_mat[source_idx];
  }
	wmma::load_matrix_sync(a_frag, dense_X_lst + warp_offset, BLK_H);

	int tcb_id_start = TCblock_rowid[bid];
	int tcb_id_end = TCblock_rowid[bid + 1];
	// for loading dense_Y
	int n_col_group = BLK_H / (WARP_SIZE / BLK_H);
	int row = tid % embedding_dim;
	int block_row_id = row % BLK_H;
	int warp_start = (row / BLK_H) * BLK_H * BLK_H;
  /////////////////////////////////
  // main loop
  /////////////////////////////////
  for (int tcb_id = tcb_id_start; tcb_id < tcb_id_end; tcb_id++) {
		#pragma unroll
    for (int idx = tid; idx < BLK_H * BLK_H; idx += threadPerBlock) {
      edge_attention_block_half[idx] = __float2half(0.0f);
			edge_attention_block_single[idx] = 0.0f;
    }

		// Initialize dense_Y by column-major store
		// Here I'm assuming embedding_dim is a multiple of BLK_H
		// and that each warp loads a BLK_H x BLK_H block
		// #pragma unroll
		for(int col_group = 0; col_group < n_col_group; col_group++){
			int block_col_id = col_group * 2 + (tid / embedding_dim);
			int X_rowId = sparse_AToX_idx[tcb_id * BLK_H + block_col_id];   
			dense_X_lst[block_col_id * BLK_H + block_row_id + warp_start] = in_mat[X_rowId * embedding_dim + row];
		}
		//also necessary
		__syncthreads();

    wmma::load_matrix_sync(b_frag, dense_X_lst + warp_offset, BLK_H);
    // clear acc_frag
    wmma::fill_fragment(acc_frag, 0.0f);
    // Perform the matrix multiplication on Tensor Core
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    // Each warp store the result into the right slice of the intermediate
    // tensor
    wmma::store_matrix_sync(sparse_A_val + warp_offset, acc_frag, BLK_H,
                            wmma::mem_row_major);

    __syncthreads();

		// Starting and ending ind of nnz of the current TCblock in TCblocktile_id
		int eIdx_start = TCblock_offset[tcb_id];
		int nnz_in_block = TCblock_offset[tcb_id + 1] - eIdx_start;
			// Add up results from all warps using cuda cores
		// NOTE: if threadPerBlock > nnz_in_block, we only need 1 iteration. 
		// But if threadPerBlock < nnz_in_block, we need multiple iterations, which could slow things down.
		for(int i = tid; i < nnz_in_block; i += threadPerBlock) {
			int eid = eIdx_start + i;
			int block_id = TCblocktile_id[eid];
			for(int j = 0; j < warpPerBlock; j++) {
				// TODO: this access pattern doesn't seem very efficient
				edge_attention_block_single[block_id] += sparse_A_val[block_id + j * BLK_H * BLK_H];
			}
			edge_attention_block_half[block_id] = __float2half(edge_attention_block_single[block_id]);
			// Save the edge attention
			// TODO: maybe we can coalesce this write
			if(save_edge_attention) {
				edgeAttention[eid] = edge_attention_block_single[block_id];
			}
		}
    // necessary to ensure edge_attention_block is correct
    __syncthreads();

    /////////
    // SpMM
    /////////
    // load the result of SDDMM
    wmma::load_matrix_sync(att_frag, edge_attention_block_half, BLK_H);
    // load feature matrix block
    wmma::load_matrix_sync(b_t_frag, dense_X_lst + warp_offset, BLK_H);

    // spmm_acc_frag might be moved in and out of registers to local(global)
    // memory every iteration, which can be bad (Ampere have 255 registers (32
    // bits each) per thread) so we have to monitor for this, which I'm not sure
    // how alternative is to compute and store the SDDMM results for all tc
    // blocks first this will need (blockDim.y * 16 * 16 * 4) bytes
    wmma::mma_sync(spmm_acc_frag, att_frag, b_t_frag, spmm_acc_frag);
  }
  wmma::store_matrix_sync(output + bid * BLK_H * embedding_dim + wid * BLK_H,
                          spmm_acc_frag, embedding_dim, wmma::mem_row_major);
}

#if defined(BLK_M) && defined(BLK_N) && defined(BLK_K) && \
    BLK_M == 8 && BLK_N == 32 && BLK_K == 16
//////////////////////
/// Same as fusedMM
/// Except the partial result for SDDMM is stored in fp32 instead of half
/// note here we are assuming only 1 attention head
//////////////////////
__global__ void TC_fusedMM_fp32_inter_m8n32k16_cuda_kernel(
		const int *__restrict__ TCblock_rowid, 		 // offset of each row window.
		const uint8_t *__restrict__ TCblocktile_id,  // id of each TC block nonzero element.
		const int *__restrict__ TCblock_offset,      // colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx,     // colid of each TC block nonzero element.
		const int numNodes, const int numEdges,
		const int embedding_dim,    // embedding dimension.
		torch::Half *__restrict__ in_mat, // input feature matrix.
		float *output,              // aggreAGNNed output feature matrix.
		float *edgeAttention, // result of SDDMM.
		bool save_edge_attention) {
	half* in_mat_half = reinterpret_cast<half*>(in_mat);
  int bid = blockIdx.x;     // block_index == row_window_index
  int wid = threadIdx.y;    // warp_index handling multi-dimension > 16.
  int laneid = threadIdx.x; // lanid of each warp.
  int tid = wid * blockDim.x + laneid; // threadid of each block.

  int warpPerBlock = blockDim.y;
  int threadPerBlock = blockDim.x * warpPerBlock;

  // starting node_id of current row_window.
  int nid_start = bid * BLK_M; 
  // ending node_id of the current row_window.
  int nid_end = min((bid + 1) * BLK_M, numNodes); 
  assert(nid_start < nid_end);

  __shared__ half edge_attention_block_half[BLK_M * BLK_N]; // Result of SDDMM
	__shared__ float edge_attention_block_single[BLK_M * BLK_N]; // Result of SDDMM in single

  extern __shared__ char dynamic_shared_mixed[]; 
  float* sparse_A_val = (float*) &dynamic_shared_mixed[0];     // result of XX^T for all warps
  // dense_X block for each warp. offset of wid*BLK_N*BLK_N is already included. 
  half* dense_X =
      (half*) &dynamic_shared_mixed[warpPerBlock * BLK_M * BLK_N * sizeof(float) + wid * BLK_N * BLK_N * sizeof(half)]; // X for all warps

  wmma::fragment<wmma::matrix_a, BLK_M, BLK_N, BLK_K, half, wmma::row_major>
      a_frag_0;
	wmma::fragment<wmma::matrix_a, BLK_M, BLK_N, BLK_K, half, wmma::row_major>
			a_frag_1;
	wmma::fragment<wmma::matrix_a, BLK_M, BLK_N, BLK_K, half, wmma::row_major>
			att_frag;
  wmma::fragment<wmma::matrix_b, BLK_M, BLK_N, BLK_K, half, wmma::col_major>
      b_frag;
  wmma::fragment<wmma::matrix_b, BLK_M, BLK_N, BLK_K, half, wmma::row_major>
      b_t_frag;
  wmma::fragment<wmma::accumulator, BLK_M, BLK_N, BLK_K, float> acc_frag;
  wmma::fragment<wmma::accumulator, BLK_M, BLK_N, BLK_K, float> spmm_acc_frag;
  wmma::fill_fragment(spmm_acc_frag, 0.0f);

  // load dense_X. each warp load 2 8x16 blocks.
	int warp_offset_X = bid * BLK_M * embedding_dim + wid * BLK_K * 2;
	wmma::load_matrix_sync(a_frag_0, in_mat_half + warp_offset_X, embedding_dim);
	wmma::load_matrix_sync(a_frag_1, in_mat_half + warp_offset_X + BLK_K, embedding_dim);

	int tcb_id_start = TCblock_rowid[bid];
	int tcb_id_end = TCblock_rowid[bid + 1];
  /////////////////////////////////
  // main loop
  /////////////////////////////////
  for (int tcb_id = tcb_id_start; tcb_id < tcb_id_end; tcb_id++) {
		#pragma unroll
    for (int idx = tid; idx < BLK_M * BLK_N; idx += threadPerBlock) {
      edge_attention_block_half[idx] = __float2half(0.0f);
			edge_attention_block_single[idx] = 0.0f;
    }

		// Each warp loads a 32x32 block of compact(X^T) in column-major order
		for(int i = 0; i < BLK_N; i ++){
			int X_rowId = sparse_AToX_idx[tcb_id * BLK_N + i];
			dense_X[i * BLK_N + laneid] = in_mat_half[X_rowId * embedding_dim + wid * BLK_N + laneid];
		}
    // print using tid == 0
    if(tid == 0 && bid == 0 && tcb_id == 0){
      for (int i = 0; i < 5; i++){
        for (int j = 0; j < 5; j++){
          printf("%f ", __half2float(dense_X[i * BLK_N + j]));
        }
        printf("\n");
      }
    }
		//also necessary
		__syncthreads();
		// __syncwarp();

    int warp_offset_S = wid * BLK_M * BLK_N;
    wmma::fill_fragment(acc_frag, 0.0f);
		for(int i = 0; i < 2; i++){
			wmma::load_matrix_sync(b_frag, dense_X + i*BLK_K, BLK_N);
			// Perform the matrix multiplication on Tensor Core
			if(i == 0)
				wmma::mma_sync(acc_frag, a_frag_0, b_frag, acc_frag);
			else
				wmma::mma_sync(acc_frag, a_frag_1, b_frag, acc_frag);
		}
    // Each warp store the result into the right slice of the intermediate
    // tensor
    wmma::store_matrix_sync(sparse_A_val + warp_offset_S, acc_frag, BLK_N,
                            wmma::mem_row_major);

    __syncthreads();

    if(tid == 0 && bid == 0 && tcb_id == 0){
      int offset = warp_offset_S + 4*BLK_N;
      printf("sparse_A_val partial warp 0:\n");
      for (int i = 0; i < 25; i++){
        printf("%f ", sparse_A_val[offset + i]);
      }
      printf("\n");

      offset = warp_offset_S + 1*BLK_M*BLK_N + 4*BLK_N;
      printf("sparse_A_val partial warp 1:\n");
      for (int i = 0; i < 25; i++){
        printf("%f ", sparse_A_val[offset + i]);
      }
      printf("\n");
    }

		// Starting and ending ind of nnz of the current TCblock in TCblocktile_id
		int eIdx_start = TCblock_offset[tcb_id];
		int nnz_in_block = TCblock_offset[tcb_id + 1] - eIdx_start;
		// Add up results from all warps using cuda cores
		// NOTE: if threadPerBlock > nnz_in_block, we only need 1 iteration. 
		// But if threadPerBlock < nnz_in_block, we need multiple iterations, which could slow things down.
		for(int i = tid; i < nnz_in_block; i += threadPerBlock) {
			int eid = eIdx_start + i;
			int block_id = TCblocktile_id[eid];
			for(int j = 0; j < warpPerBlock; j++) {
				edge_attention_block_single[block_id] += sparse_A_val[block_id + j * BLK_M * BLK_N];
			}
			edge_attention_block_half[block_id] = __float2half(edge_attention_block_single[block_id]);
			// Save the edge attention
			if(save_edge_attention) {
				edgeAttention[eid] = edge_attention_block_single[block_id];
			}
		}
    // necessary to ensure edge_attention_block is correct
    __syncthreads();

    /////////
    // SpMM
    /////////
		for(int i = 0; i < 2; i++){
			// load the result of SDDMM
			wmma::load_matrix_sync(att_frag, edge_attention_block_half + i*BLK_K, BLK_N);
			// load feature matrix block
			wmma::load_matrix_sync(b_t_frag, dense_X + i*BLK_K*BLK_N, BLK_N);
			// spmm_acc_frag might be moved in and out of registers to local(global)
			// memory every iteration, which can be bad (Ampere have 255 registers (32
			// bits each) per thread) so we have to monitor for this, which I'm not sure
			// how alternative is to compute and store the SDDMM results for all tc
			// blocks first this will need (blockDim.y * 16 * 16 * 4) bytes
			wmma::mma_sync(spmm_acc_frag, att_frag, b_t_frag, spmm_acc_frag);
		}
  }
  wmma::store_matrix_sync(output + bid * BLK_M * embedding_dim + wid * BLK_N,
                          spmm_acc_frag, embedding_dim, wmma::mem_row_major);
}
#endif

#if defined(BLK_M) && defined(BLK_N) && defined(BLK_K) && \
    BLK_M == 16 && BLK_N == 8 && BLK_K == 16
__global__ void f3s_m16n8k16_cuda_kernel(
		const int *__restrict__ TCblock_rowid, 		 // offset of each row window.
		const int *__restrict__ sparse_AToX_idx,     // colid of each TC block nonzero element.
    const uint64_t *__restrict__ TCblock_bit_map,
    const int numNodes,
    const int embedding_dim,
		torch::Half *__restrict__ Q, 
    torch::Half *__restrict__ K, 
    torch::Half *__restrict__ V,
		float *output,
		float *sddmm_result
    ){
  // grouped by threads. E.g. first blockDim.y * 8 values stores the D_frag of thread 0 of each warp.
  extern __shared__ float partial_sum[];
  // 2 16x8 blocks, each block is divided into 2 8x8 subblocks in row major order.
  __shared__ float sum[BLK_M * BLK_N * 2];

  int bid = blockIdx.x;     // block_index == row_window_index
  int wid = threadIdx.y;    // warp_index handling multi-dimension > 16.
  int laneid = threadIdx.x; // lanid of each warp.
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  // int warp_offset = wid * BLK_M * BLK_N * 2; // offset for partial_sum

  half2* K_half2 = reinterpret_cast<half2*>(K); 
  half2* Q_half2 = reinterpret_cast<half2*>(Q);

  // starting node_id of current row_window.
  int nid_start = bid * BLK_H; 
  // ending node_id of the current row_window.
  int nid_end = min((bid + 1) * BLK_H, numNodes); 
  assert(nid_start < nid_end);
 
  uint32_t Q_frag[4];
  uint32_t B_frag[2];
  float D_frag[8];// sddmm intermediate
  uint32_t S_frag[4];// sddmm result
  float O_frag[8] = {0};// spmm result
  float C_frag[4] = {0};
  
  for(int i = tid; i < BLK_M * BLK_N * 2; i += blockDim.x * blockDim.y){
    sum[i] = 0.0f;
  }
  // Threads of a warp for fetching a 16X16 block of Q.
  // DOC: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=wmma#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
  // Here I'm swapping columns of Q to make the memory access more coalesced. 
  // So when loading K, we have to swap the rows accordingly in order to get the same result.
  int rowIdx = bid * BLK_M + laneid/4;
  // /2 because half2. *2 because reading 2 consecutive half2. 
  int colIdx = wid * BLK_K/2 + (laneid%4) * 2;
  half2_uint32 h2U32Converter;
  h2U32Converter.h2 = Q_half2[rowIdx * embedding_dim/2 + colIdx];
  Q_frag[0] = h2U32Converter.u32;
  h2U32Converter.h2 = Q_half2[rowIdx * embedding_dim/2 + colIdx + 1];
  Q_frag[2] = h2U32Converter.u32;
  h2U32Converter.h2 = Q_half2[(rowIdx+8) * embedding_dim/2 + colIdx];
  Q_frag[1] = h2U32Converter.u32;
  h2U32Converter.h2 = Q_half2[(rowIdx+8) * embedding_dim/2 + colIdx + 1];
  Q_frag[3] = h2U32Converter.u32;

	int tcb_id_start = TCblock_rowid[bid];
	int tcb_id_end = TCblock_rowid[bid + 1];
  int laneid_rev = 63 - laneid*2;
  /////////////////////////////////
  // main loop
  /////////////////////////////////
  bool odd_number_of_blocks = (tcb_id_end - tcb_id_start) % 2;
  bool last_block = false;
  for (int tcb_id = tcb_id_start; tcb_id < tcb_id_end; tcb_id+=2) {
    if(tcb_id == tcb_id_end - 1 && odd_number_of_blocks){
      last_block = true;
    }
		// Initialize B_frag from K
    // Note I'm swapping rows of B_frag because we swapped the columns of A_frag(Q)
		// Assuming embedding_dim is a multiple of BLK_H
		// loop over 2 column major blocks in B_frag
    // index in terms of half2, only affect rowIdx
    // /2 because half2. *2 because reading 2 consecutive half2. 
    // +i instead of +i*4 because of the col swap
    colIdx = (wid * BLK_M)/2 + (laneid % 4)*2; 
    for(int i = 0; i < 2; i++){
      if(!last_block || i == 0){
        rowIdx = sparse_AToX_idx[(tcb_id+i) * BLK_N + laneid / 4]; 
        h2U32Converter.h2 = K_half2[rowIdx * embedding_dim/2 + colIdx];
        B_frag[0] = h2U32Converter.u32;
        h2U32Converter.h2 = K_half2[rowIdx * embedding_dim/2 + colIdx + 1];
        B_frag[1] = h2U32Converter.u32;
        HMMA16816(D_frag[i*4+0], D_frag[i*4+1], D_frag[i*4+2], D_frag[i*4+3], 
                  Q_frag[0], Q_frag[1], Q_frag[2], Q_frag[3], 
                  B_frag[0], B_frag[1], 
                  C_frag[0], C_frag[1], C_frag[2], C_frag[3]);
      }
    }
    __syncthreads();
    for(int i =0; i< 2; i++){// 2 16x8 blocks
      if(!last_block || i == 0){
        int sum_offset = i*BLK_M*BLK_N;
        for(int j=0; j< 2; j++){// 2 8x8 blocks in each 16x8 block
          if((TCblock_bit_map[(tcb_id+i)*2+j] & (1ULL << laneid_rev)) != 0){
            atomicAdd(&sum[sum_offset + j*BLK_N*BLK_N + laneid*2], D_frag[i*4 + j*2]);
          }
          if((TCblock_bit_map[(tcb_id+i)*2+j] & (1ULL << (laneid_rev-1))) != 0){
            atomicAdd(&sum[sum_offset + j*BLK_N*BLK_N + laneid*2 + 1], D_frag[i*4 + j*2 + 1]);
          }
        }
      }
    }
    __syncthreads();
    if(sddmm_result != nullptr){
      // have warp 0 load sum into sddmm_result, which is row major
      if(wid == 0){
        int offset = tcb_id * BLK_M * BLK_N;
        for(int i = 0; i < 2; i++){ // 2 16x8 blocks
          for(int j = 0; j < 2; j++){ // 2 8x8 blocks in each 16x8 block
            int sum_offset = i*BLK_M*BLK_N + j*BLK_N*BLK_N + laneid*2;
            sddmm_result[offset + sum_offset] = sum[sum_offset];
            sddmm_result[offset + sum_offset + 1] = sum[sum_offset + 1];
          }
        }
      }
    }
    float2* sum_float2 = reinterpret_cast<float2*>(sum);
    for(int i = 0; i < 2; i++){// 2 16x8 blocks
      int sum_offset = i*BLK_M*BLK_N/2;
      for(int j = 0; j < 2; j++){// 2 8x8 blocks in each 16x8 block
        float2 temp = sum_float2[sum_offset + j*BLK_N*BLK_N/2 + laneid];
        h2U32Converter.h2.x = __float2half(temp.x);
        h2U32Converter.h2.y = __float2half(temp.y);
        S_frag[i*2+j] = h2U32Converter.u32;
      }
    }
    __syncthreads();
    //reset sum to 0
    for(int i = tid; i < BLK_M * BLK_N * 2; i += blockDim.x * blockDim.y){
      sum[i] = 0.0f;
    }
    /////////
    // SpMM
    /////////
    for(int j = 0; j < 2; j++){// 2 16x8 blocks
      half temp_V[4];
      int colIdx = (wid*2+j) * BLK_N + laneid/4;
      if(!last_block || j == 0){
        for(int i = 0; i < 2; i++){// 2 8x8 blocks in each 16x8 block
          for(int k = 0; k < 2; k++){// 2 halfs in each 8x8 block
            int rowIdx = sparse_AToX_idx[tcb_id * BLK_N + (laneid%4)*2 + i*8 + k];
            temp_V[i*2 + k] = V[rowIdx * embedding_dim + colIdx];
          }
        }
        h2U32Converter.h2 = __halves2half2(temp_V[0], temp_V[1]);
        B_frag[0] = h2U32Converter.u32;
        h2U32Converter.h2 = __halves2half2(temp_V[2], temp_V[3]);
        B_frag[1] = h2U32Converter.u32;
      }
      else{
        B_frag[0] = 0;
        B_frag[1] = 0;
      }
      HMMA16816(O_frag[4*j], O_frag[4*j+1], O_frag[4*j+2], O_frag[4*j+3], 
                S_frag[0], S_frag[1], S_frag[2], S_frag[3], 
                B_frag[0], B_frag[1], 
                O_frag[4*j], O_frag[4*j+1], O_frag[4*j+2], O_frag[4*j+3]);
    }
  }
  for(int j=0; j < 2; j++){// 2 8x8 blocks in each 16x8 block
    int rowIdx = bid * BLK_M + (laneid / 4) + j * BLK_M/2;
    for(int i =0; i < 2; i++){// 2 16x8 blocks
      int colIdx = (wid * 2 + i) * BLK_N + (laneid % 4) * 2;
      output[rowIdx * embedding_dim + colIdx] = O_frag[i*4 + j*2];
      output[rowIdx * embedding_dim + colIdx + 1] = O_frag[i*4 + j*2 + 1]; 
    }
  }
}
#endif