#include "config.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <mma.h>
#include <sstream>
#include <stdio.h>
// #include <thrust/device_vector.h>
// #include <thrust/host_vector.h>
// #include <thrust/sort.h>
// #include <thrust/unique.h>
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

__device__ void save_sddmm_result(float* sum, float* sddmm_result, int tcb_id, int last_block){
  int wid = threadIdx.y;    // warp_index handling multi-dimension > 16.
  int laneid = threadIdx.x; // lanid of each warp.
  // have warp 0 load sum into sddmm_result, which is block row major
  if(wid == 0){
    int offset = tcb_id * BLK_M * BLK_N;
    for(int i = 0; i < 2; i++){ // 2 16x8 blocks
      if(!last_block || i == 0){
        for(int j = 0; j < 2; j++){ // 2 8x8 blocks in each 16x8 block
          int sum_offset = i*BLK_M*BLK_N + j*BLK_N*BLK_N + laneid*2;
          sddmm_result[offset + sum_offset] = sum[sum_offset];
          sddmm_result[offset + sum_offset + 1] = sum[sum_offset + 1];
        }
      }
    }
  }
}

// produce the max value for every subwarp of 4 threads
__device__ void reduce_max(float& max, int lane_id){
  int group_id = lane_id / 4;
  // Mask for subwarp of 4 threads
  unsigned int group_mask = 0xF << (group_id * 4);

  // Perform comparisons in two rounds
  float neighbor = __shfl_xor_sync(group_mask, max, 1, 4);
  max = fmaxf(max, neighbor);
  neighbor = __shfl_xor_sync(group_mask, max, 2, 4);
  max = fmaxf(max, neighbor);
}

// sum up "sum" for every 4 consecutive threads in a warp
// results is only valid for the first thread in the warp
__device__ void reduce_sum(float& sum){
  for (int offset = 1; offset < 4; offset *= 2) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset, 4);
  }
}

__device__ void sum_warp(const uint64_t* TCblock_bit_map, float* sum, float* D_frag, int tcb_id, bool last_block, int laneid){
  uint64_t bit_mask = 1ULL << (63 - laneid*2);
  uint64_t bit_mask_next = 1ULL << (63 - laneid*2-1);
  for(int i =0; i< 2; i++){// 2 16x8 blocks
    if(!last_block || i == 0){
      int sum_offset = i*BLK_M*BLK_N;
      for(int j=0; j< 2; j++){// 2 8x8 blocks in each 16x8 block
        if((TCblock_bit_map[(tcb_id+i)*2+j] & bit_mask) != 0){
          atomicAdd(&sum[sum_offset + j*BLK_N*BLK_N + laneid*2], D_frag[i*4 + j*2]);
        }
        if((TCblock_bit_map[(tcb_id+i)*2+j] & bit_mask_next) != 0){
          atomicAdd(&sum[sum_offset + j*BLK_N*BLK_N + laneid*2 + 1], D_frag[i*4 + j*2 + 1]);
        }
      }
    }
  }
}

__device__ void sum_warp_i(int i, const uint64_t* TCblock_bit_map, float* sum, float* D_frag, int tcb_id, bool last_block, int laneid){
  uint64_t bit_mask = 1ULL << (63 - laneid*2);
  uint64_t bit_mask_next = 1ULL << (63 - laneid*2-1);
  if(!last_block || i == 0){
    int sum_offset = i*BLK_M*BLK_N;
    for(int j=0; j< 2; j++){// 2 8x8 blocks in each 16x8 block
      if((TCblock_bit_map[(tcb_id+i)*2+j] & bit_mask) != 0){
        atomicAdd(&sum[sum_offset + j*BLK_N*BLK_N + laneid*2], D_frag[j*2]);
      }
      if((TCblock_bit_map[(tcb_id+i)*2+j] & bit_mask_next) != 0){
        atomicAdd(&sum[sum_offset + j*BLK_N*BLK_N + laneid*2 + 1], D_frag[j*2 + 1]);
      }
    }
  }
}

//sum should be BLK_M*BLK_M*number of warps.
__device__ void store_sum_shm(int i, float* sum, float* D_frag, int wid, int laneid){
  int sum_offset = wid*BLK_M*BLK_M + i*BLK_M*BLK_N + laneid*2;
  sum[sum_offset] = D_frag[0];
  sum[sum_offset + 1] = D_frag[1];
  sum[sum_offset + BLK_N*BLK_N] = D_frag[2];
  sum[sum_offset + BLK_N*BLK_N + 1] = D_frag[3];
}

__device__ void sum_partial_sum(float* sum, int tcb_id, int tid, int n_warps, const uint64_t* TCblock_bit_map, bool last_block){
  for(int ind = tid; ind < BLK_M*BLK_M; ind += blockDim.x*blockDim.y){
    int block_id = ind / 64; // which 8x8 block ind belongs to
    if(!last_block || block_id < 2){
      int block_offset = ind % 64; // which element in the 8x8 block ind belongs to
      uint64_t bit_mask = 1ULL << (63 - block_offset);
      if((TCblock_bit_map[tcb_id*2+block_id] & bit_mask) != 0){
        //skip first warp because it's already loaded into sum
        for(int i = 1; i < n_warps; i++){
          sum[ind] += sum[i*BLK_M*BLK_M + ind];
        }
      }
      else{
        sum[ind] = 0.0f;
      }
    }
    else{
      sum[ind] = 0.0f;
    }
  }
}

__device__ void set_Q_frag_uint64(volatile uint32_t* Q_frag, uint64_t* Q, int bid, int wid, int laneid, int numNodes, int embedding_dim){
   // Threads of a warp for fetching a 16X16 block of Q.
  // DOC: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=wmma#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
  // Here I'm swapping columns of Q to make the memory access more coalesced. 
  // So when loading K, we have to swap the rows accordingly in order to get the same result.
  int rowIdx = bid * BLK_M + laneid/4;
  // /4 because half4.
  int colIdx = wid * BLK_K/4 + (laneid%4);
  if(rowIdx < numNodes){
    uint64_t val = Q[rowIdx * embedding_dim/4 + colIdx];
    Q_frag[0] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
    Q_frag[2] = static_cast<uint32_t>(val >> 32);
  }
  if(rowIdx + 8 < numNodes){
    uint64_t val = Q[(rowIdx+8) * embedding_dim/4 + colIdx];
    Q_frag[1] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
    Q_frag[3] = static_cast<uint32_t>(val >> 32);
  }
}

__device__ void set_Q_frag_uint64_shm(float* dyn_shm, uint64_t* Q, int bid, int wid, int laneid, int numNodes, int embedding_dim){
  uint32_t* Q_frag = reinterpret_cast<uint32_t*>(dyn_shm + BLK_M*2 + wid*blockDim.x*4);
  int rowIdx = bid * BLK_M + laneid/4;
  // /4 because half4.
  int colIdx = wid * BLK_K/4 + (laneid%4);
  if(rowIdx < numNodes){
    uint64_t val = Q[rowIdx * embedding_dim/4 + colIdx];
    Q_frag[laneid] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
    Q_frag[blockDim.x*2 + laneid] = static_cast<uint32_t>(val >> 32);
  }
  if(rowIdx + 8 < numNodes){
    uint64_t val = Q[(rowIdx+8) * embedding_dim/4 + colIdx];
    Q_frag[blockDim.x + laneid] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
    Q_frag[blockDim.x*3 + laneid] = static_cast<uint32_t>(val >> 32);
  }
}

__device__ void print_sum(float* sum, int bid, int wid, int laneid){
  if(bid == 0 && wid == 0 && laneid == 0){
    printf("sum: \n");
    for(int i = 0; i < 2; i++){
      for(int j = 0; j < 2; j++){
        for(int k = 0; k < BLK_N; k++){
          for(int l = 0; l < BLK_N; l++){
            printf("%f, ", sum[i*BLK_M*BLK_N + j*BLK_N*BLK_N + k*BLK_N + l]);
          }
          printf("\n");
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
}

// __global__ void TC_fusedMM_cuda_kernel(
// 	const int *__restrict__ TCblock_rowid, 		 // offset of each row window.
// 	const uint8_t *__restrict__ TCblocktile_id,  // id of each TC block nonzero element.
// 	const int *__restrict__ TCblock_offset,      // colid of each TC block nonzero element.
// 	const int *__restrict__ sparse_AToX_idx,     // colid of each TC block nonzero element.
// 	const int numNodes, const int numEdges,
// 	const int embedding_dim,    // embedding dimension.
// 	float *__restrict__ in_mat, // input feature matrix.
// 	float *output,              // aggreAGNNed output feature matrix.
// 	torch::Half *edgeAttention, // result of SDDMM.
// 	bool save_edge_attention
// );
// __global__ void TC_fusedMM_fp32_inter_cuda_kernel(
// 	const int *__restrict__ TCblock_rowid, 		 // offset of each row window.
// 	const uint8_t *__restrict__ TCblocktile_id,  // id of each TC block nonzero element.
// 	const int *__restrict__ TCblock_offset,      // colid of each TC block nonzero element.
// 	const int *__restrict__ sparse_AToX_idx,     // colid of each TC block nonzero element.
// 	const int numNodes, const int numEdges,
// 	const int embedding_dim,    // embedding dimension.
// 	torch::Half *__restrict__ in_mat, // input feature matrix.
// 	float *output,              // aggreAGNNed output feature matrix.
// 	float *edgeAttention, // result of SDDMM.
// 	bool save_edge_attention
// );
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
		float *sddmm_result, // result of SDDMM
    bool apply_softmax
);

__global__ void sddmm_kernel_1tb1rw(
    const int *__restrict__ RowWindowOffset,
    const int *__restrict__ TCblockRowid,
    const int *__restrict__ SparseAToXidx, 
    const uint64_t *__restrict__ TCblockBitMap,
    int embeddingDim,
    torch::Half *__restrict__ Q, 
    torch::Half *__restrict__ K, 
    float2 *output);

__global__ void sddmm_kernel_1tbnrw(
  const int *__restrict__ RowWindowOffset,
  const int *__restrict__ TBBoundaries,
  const int *__restrict__ TCblockRowid,
  const int *__restrict__ SparseAToXidx, 
  const uint64_t *__restrict__ TCblockBitMap,
  int embeddingDim,
  torch::Half *__restrict__ Q, 
  torch::Half *__restrict__ K, 
  float2 *output);

std::vector<torch::Tensor> f3S_forward_cuda(
    torch::Tensor Rowwindow_offset,
    torch::Tensor sparse_AToX_idx, 
    torch::Tensor TCblock_bit_map,
    int num_nodes, 
    int embedding_dim,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, 
    bool apply_softmax,
    bool save_sddmm_result){
  int nBlockEmbeddingDim = (embedding_dim + BLK_M - 1) / BLK_M;
  int nWarpPerBlock = nBlockEmbeddingDim;
  const int nRowWindow = Rowwindow_offset.size(0) - 1;
  int paddedLength = nRowWindow * BLK_M;
  auto output = torch::zeros({paddedLength, embedding_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  dim3 grid(nRowWindow, 1, 1);
  dim3 block(WARP_SIZE, nWarpPerBlock, 1);
  int nTCBlock = sparse_AToX_idx.size(0)/BLK_N;
	torch::Tensor sddmm_result = torch::zeros({nTCBlock*BLK_M*BLK_N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  float* sddmm_result_ptr = save_sddmm_result ? sddmm_result.data_ptr<float>() : nullptr;
  // int fixed_shared_size = 12 * WARP_SIZE * nWarpPerBlock * sizeof(float);
  int fixed_shared_size = nWarpPerBlock * BLK_M * BLK_M * sizeof(float);
  int dynamic_shared_size = apply_softmax ? fixed_shared_size + 2 * BLK_M * sizeof(float) : fixed_shared_size;
  #if BLK_M == 16 && BLK_N == 8 && BLK_K == 16
  f3s_m16n8k16_cuda_kernel<<<grid, block, dynamic_shared_size>>>(
    Rowwindow_offset.data_ptr<int>(), 
    sparse_AToX_idx.data_ptr<int>(),
    TCblock_bit_map.data_ptr<uint64_t>(),
    num_nodes, embedding_dim,
    Q.data_ptr<torch::Half>(), 
    K.data_ptr<torch::Half>(), 
    V.data_ptr<torch::Half>(),
    output.data_ptr<float>(),
    sddmm_result_ptr,
    apply_softmax);
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

std::vector<torch::Tensor> 
f3S_sddmm_cuda_1tb1rw(
    torch::Tensor RowWindowOffset,
    torch::Tensor TCblockRowid,
    torch::Tensor SparseAToXidx,
    torch::Tensor TCblockBitMap,
    int nNodes,
    int embeddingDim,
    torch::Tensor Q, torch::Tensor K,
    int nWarpPerBlock){
  int nRowWindow = RowWindowOffset.size(0) - 1;
  int nTCBlock = SparseAToXidx.size(0)/BLK_N;
  torch::Tensor sddmmResult = torch::zeros({nTCBlock*BLK_M*BLK_N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));  
  int shared_size = BLK_M * embeddingDim * sizeof(half);
  dim3 grid(nRowWindow, 1, 1);
  dim3 block(WARP_SIZE, nWarpPerBlock, 1);
  sddmm_kernel_1tb1rw<<<grid, block, shared_size>>>(
    RowWindowOffset.data_ptr<int>(), 
    TCblockRowid.data_ptr<int>(),
    SparseAToXidx.data_ptr<int>(),
    TCblockBitMap.data_ptr<uint64_t>(),
    embeddingDim,
    Q.data_ptr<torch::Half>(), 
    K.data_ptr<torch::Half>(), 
    reinterpret_cast<float2*>(sddmmResult.data_ptr<float>()));
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
  return {sddmmResult};
}

std::vector<torch::Tensor> 
f3S_sddmm_cuda_1tbnrw(
    torch::Tensor RowWindowOffset,
    torch::Tensor TBBoundaries,
    torch::Tensor TCblockRowid,
    torch::Tensor SparseAToXidx,
    torch::Tensor TCblockBitMap,
    int nNodes,
    int embeddingDim,
    torch::Tensor Q, torch::Tensor K,
    int nWarpPerBlock){
  int nRowWindow = TBBoundaries.size(0) - 1;
  int nTCBlock = SparseAToXidx.size(0)/BLK_N;
  torch::Tensor sddmmResult = torch::zeros({nTCBlock*BLK_M*BLK_N}, 
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));  
  dim3 grid(nRowWindow, 1, 1);
  dim3 block(WARP_SIZE, nWarpPerBlock, 1);
  sddmm_kernel_1tbnrw<<<grid, block>>>(
    RowWindowOffset.data_ptr<int>(), 
    TBBoundaries.data_ptr<int>(),
    TCblockRowid.data_ptr<int>(),
    SparseAToXidx.data_ptr<int>(),
    TCblockBitMap.data_ptr<uint64_t>(),
    embeddingDim,
    Q.data_ptr<torch::Half>(), 
    K.data_ptr<torch::Half>(), 
    reinterpret_cast<float2*>(sddmmResult.data_ptr<float>()));
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  return {sddmmResult};
}



// #if defined(BLK_M) && defined(BLK_N) && defined(BLK_K) && \
//     BLK_M == 16 && BLK_N == 8 && BLK_K == 16
#define bid blockIdx.x
// #define wid threadIdx.y
// #define laneid threadIdx.x
#define tid (threadIdx.x + threadIdx.y * blockDim.x)
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
		float *sddmm_result,
    bool apply_softmax){
  volatile int wid = threadIdx.y;    // warp_index handling multi-dimension > 16.
  volatile int laneid = threadIdx.x; // lanid of each warp.

  uint64_t* K_uint64 = reinterpret_cast<uint64_t*>(K);

  // row_max, row_sum (size BLK_M each) for online-softmax,
  // then Q_frag, then O_frag
  extern __shared__ float dyn_shm[]; 
  for(int i = tid; i < BLK_M*2+blockDim.y*BLK_M*BLK_M; i += blockDim.x*blockDim.y){
  // for(int i = tid; i < BLK_M*2; i += blockDim.x*blockDim.y){
    dyn_shm[i] = 0.0f;
  }
 
  float* sum = dyn_shm + BLK_M*2;
  float O_frag[8] = {0};// spmm result
  uint32_t Q_frag[4] = {0};
  set_Q_frag_uint64(Q_frag, reinterpret_cast<uint64_t*>(Q), bid, wid, laneid, numNodes, embedding_dim);
  // set_Q_frag_uint64_shm(dyn_shm, reinterpret_cast<uint64_t*>(Q), bid, wid, laneid, numNodes, embedding_dim);

  /////////////////////////////////
  // main loop
  /////////////////////////////////
  volatile bool last_block = false;
  for (int tcb_id = TCblock_rowid[bid]; tcb_id < TCblock_rowid[bid + 1]; tcb_id+=2) {
    if((TCblock_rowid[bid + 1] - TCblock_rowid[bid]) % 2 && tcb_id == TCblock_rowid[bid + 1] - 1){
      last_block = true;
    }
    {
      uint32_t B_frag[2];
      float D_frag[4];
      // uint32_t* Q_frag = reinterpret_cast<uint32_t*>(dyn_shm + BLK_M*2 + wid*blockDim.x*4);
      int colIdx = (wid * BLK_M)/4 + (laneid % 4); 
      for(int i = 0; i < 2; i++){
        if(!last_block || i == 0){
          // Initialize B_frag from K
          // Note I'm swapping rows of B_frag because we swapped the columns of A_frag(Q)
          // index in terms of half2, only affect rowIdx
          int rowIdx = sparse_AToX_idx[(tcb_id+i) * BLK_N + laneid / 4]; 
          uint64_t val = K_uint64[rowIdx * embedding_dim/4 + colIdx];
          B_frag[0] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
          B_frag[1] = static_cast<uint32_t>(val >> 32);
          HMMA16816(D_frag[0], D_frag[1], D_frag[2], D_frag[3], 
                    // Q_frag[laneid], Q_frag[blockDim.x + laneid], Q_frag[blockDim.x*2 + laneid], Q_frag[blockDim.x*3 + laneid], 
                    Q_frag[0], Q_frag[1], Q_frag[2], Q_frag[3], 
                    B_frag[0], B_frag[1], 
                    0.0f, 0.0f, 0.0f, 0.0f);
          store_sum_shm(i, sum, D_frag, wid, laneid);
        }
      }
      __syncthreads();
      sum_partial_sum(sum, tcb_id, tid, blockDim.y, TCblock_bit_map, last_block);
    }
    __syncthreads();

    if(sddmm_result != nullptr){
      save_sddmm_result(sum, sddmm_result, tcb_id, last_block);
    }

    // {// softmax + spmm
    //   uint32_t S_frag[4];// softmax/sddmm result
    //   if(apply_softmax){
    //     float D_frag[4];
    //     float2* sum_float2 = reinterpret_cast<float2*>(sum);
    //     for(int j = 0; j < 2; j++){// 2 8x8 blocks in each 16x8 block
    //       int sum_offset = j*BLK_N*BLK_N/2;
    //       for(int i = 0; i < 2; i++){// 2 16x8 blocks
    //         if(!last_block || i == 0){
    //           float2 temp = sum_float2[i*BLK_M*BLK_N/2 + sum_offset + laneid];
    //           D_frag[i*2] = temp.x;
    //           D_frag[i*2 + 1] = temp.y;
    //         }
    //         else{
    //           D_frag[i*2] = 0.0f;
    //           D_frag[i*2 + 1] = 0.0f;
    //         }
    //       }
    //       //max of the 4 elements in the same row across 2 16x8 blocks
    //       //need every warp to do this because they will need it for the next computation
    //       float max_old = dyn_shm[j*BLK_N + laneid/4];

    //       float max = fmaxf(
    //         fmaxf(fmaxf(D_frag[0], D_frag[1]), fmaxf(D_frag[2], D_frag[3])), 
    //         max_old);
    //       reduce_max(max, laneid);

    //       for(int i = 0; i < 4; i++){
    //         if(D_frag[i] != 0.0f){
    //           D_frag[i] = __expf(D_frag[i] - max);
    //         }
    //       }

    //       float exp_max_diff = __expf(max_old - max);

    //       if(wid == 0){
    //         float sum = D_frag[0] + D_frag[1] + D_frag[2] + D_frag[3];
    //         reduce_sum(sum);
    //         if(laneid % 4 == 0){
    //           dyn_shm[BLK_M + j*BLK_N + laneid/4] = dyn_shm[BLK_M + j*BLK_N + laneid/4] * exp_max_diff + sum;
    //         }
    //       }

    //       O_frag[j*2]   = O_frag[j*2]   * exp_max_diff;
    //       O_frag[j*2+1] = O_frag[j*2+1] * exp_max_diff;
    //       O_frag[j*2+4] = O_frag[j*2+4] * exp_max_diff;
    //       O_frag[j*2+5] = O_frag[j*2+5] * exp_max_diff;
    //       // float* O_frag = dyn_shm + BLK_M*2 + blockDim.x*blockDim.y*4 + wid*blockDim.x*8;
    //       // O_frag[blockDim.x*j*2 + laneid]     = O_frag[blockDim.x*j*2 + laneid]   * (exp_max_diff);
    //       // O_frag[blockDim.x*(j*2+1) + laneid] = O_frag[blockDim.x*(j*2+1) + laneid] * (exp_max_diff);
    //       // O_frag[blockDim.x*(j*2+4) + laneid] = O_frag[blockDim.x*(j*2+4) + laneid] * (exp_max_diff);
    //       // O_frag[blockDim.x*(j*2+5) + laneid] = O_frag[blockDim.x*(j*2+5) + laneid] * (exp_max_diff);

    //       if(wid == 0 && laneid % 4 == 0){
    //         dyn_shm[j*BLK_N + laneid/4] = max;
    //       }

    //       half2_uint32 h2U32Converter;
    //       for(int i = 0; i < 2; i++){
    //         h2U32Converter.h2.x = __float2half(D_frag[i*2]);
    //         h2U32Converter.h2.y = __float2half(D_frag[i*2+1]);
    //         S_frag[i*2 + j] = h2U32Converter.u32;
    //       }
    //     }
    //   }
    //   else{
    //     float2* sum_float2 = reinterpret_cast<float2*>(sum);
    //     for(int i = 0; i < 2; i++){// 2 16x8 blocks
    //       int sum_offset = i*BLK_M*BLK_N/2;
    //       half2_uint32 h2U32Converter;
    //       if(!last_block || i == 0){
    //         for(int j = 0; j < 2; j++){// 2 8x8 blocks in each 16x8 block
    //           float2 temp = sum_float2[sum_offset + j*BLK_N*BLK_N/2 + laneid];
    //           h2U32Converter.h2.x = __float2half(temp.x);
    //           h2U32Converter.h2.y = __float2half(temp.y);
    //           S_frag[i*2+j] = h2U32Converter.u32;
    //         }
    //       }
    //       else{
    //         S_frag[i*2] = 0;
    //         S_frag[i*2+1] = 0;
    //       }
    //     }
    //   }
    //   __syncthreads();
    //   //reset sum to 0
    //   // for(int i = tid; i < BLK_M * BLK_N * 2; i += blockDim.x * blockDim.y){
    //   //   sum[i] = 0.0f;
    //   // }
    //   for(int i = laneid; i < BLK_M*BLK_M; i += blockDim.x){
    //     sum[wid*BLK_M*BLK_M + i] = 0.0f;
    //   }
    //   /////////
    //   // SpMM
    //   /////////
    //   {
    //     uint32_t B_frag[2];
    //     half2_uint32 h2U32Converter;
    //     half temp_V[2];
    //     // float* O_frag = dyn_shm + BLK_M*2 + blockDim.x*blockDim.y*4 + wid*blockDim.x*8;
    //     for(int j = 0; j < 2; j++){// 2 16x8 blocks
    //       int colIdx = (wid*2+j) * BLK_N + laneid/4;
    //       for(int i = 0; i < 2; i++){// 2 8x8 blocks in each 16x8 block
    //         if(!last_block || i == 0){
    //           for(int k = 0; k < 2; k++){// 2 halfs in each 8x8 block
    //             int rowIdx = sparse_AToX_idx[(tcb_id+i) * BLK_N + (laneid%4)*2 + k];
    //             temp_V[k] = V[rowIdx * embedding_dim + colIdx];
    //           }
    //           h2U32Converter.h2 = __halves2half2(temp_V[0], temp_V[1]);
    //           B_frag[i] = h2U32Converter.u32;
    //         }
    //         else{
    //           B_frag[i] = 0;
    //         }
    //       }
    //       // HMMA16816(O_frag[blockDim.x*j*4 + laneid], O_frag[blockDim.x*(j*4+1) + laneid], O_frag[blockDim.x*(j*4+2) + laneid], O_frag[blockDim.x*(j*4+3) + laneid], 
    //       HMMA16816(O_frag[4*j], O_frag[4*j+1], O_frag[4*j+2], O_frag[4*j+3],
    //                 S_frag[0], S_frag[1], S_frag[2], S_frag[3], 
    //                 B_frag[0], B_frag[1], 
    //                 // O_frag[blockDim.x*4*j + laneid], O_frag[blockDim.x*(4*j+1) + laneid], O_frag[blockDim.x*(4*j+2) + laneid], O_frag[blockDim.x*(4*j+3) + laneid]);
    //                 O_frag[4*j], O_frag[4*j+1], O_frag[4*j+2], O_frag[4*j+3]);
    //     }
    //   }
    // }
  }
  // // float* O_frag = dyn_shm + BLK_M*2 + blockDim.x*blockDim.y*4 + wid*blockDim.x*8;
  // if(apply_softmax){
  //   for(int i = 0; i < 2; i++){
  //     float row_sum = dyn_shm[BLK_M + laneid/4 + i*BLK_N ];
  //     if(row_sum != 0.0f){
  //       // O_frag[blockDim.x*i*2 + laneid] = O_frag[blockDim.x*i*2 + laneid] * (1.0f/row_sum);
  //       // O_frag[blockDim.x*(i*2+1) + laneid] = O_frag[blockDim.x*(i*2+1) + laneid] * (1.0f/row_sum);
  //       // O_frag[blockDim.x*(i*2+4) + laneid] = O_frag[blockDim.x*(i*2+4) + laneid] * (1.0f/row_sum);
  //       // O_frag[blockDim.x*(i*2+5) + laneid] = O_frag[blockDim.x*(i*2+5) + laneid] * (1.0f/row_sum);
  //       O_frag[i*2] = O_frag[i*2] * (1.0f/row_sum);
  //       O_frag[(i*2+1)] = O_frag[(i*2+1)] * (1.0f/row_sum);
  //       O_frag[(i*2+4)] = O_frag[(i*2+4)] * (1.0f/row_sum);
  //       O_frag[(i*2+5)] = O_frag[(i*2+5)] * (1.0f/row_sum);
        
  //     }
  //   }
  // }
  for(int j=0; j < 2; j++){// 2 8x8 blocks in each 16x8 block
    int rowIdx = bid * BLK_M + (laneid / 4) + j * BLK_M/2;
    for(int i =0; i < 2; i++){// 2 16x8 blocks
      int colIdx = (wid * 2 + i) * BLK_N + (laneid % 4) * 2;
      // output[rowIdx * embedding_dim + colIdx] = O_frag[blockDim.x*(i*4 + j*2) + laneid];
      // output[rowIdx * embedding_dim + colIdx + 1] = O_frag[blockDim.x*(i*4 + j*2 + 1) + laneid]; 
      output[rowIdx * embedding_dim + colIdx] = O_frag[i*4 + j*2];
      output[rowIdx * embedding_dim + colIdx + 1] = O_frag[i*4 + j*2 + 1]; 
    }
  }
}

__device__ void load_K_frag_permute_col(volatile uint32_t *K_frag, uint64_t *__restrict__ K, int embedding_dim, int rowIdx, int colIdx) {
  uint64_t val = K[rowIdx + colIdx];
  K_frag[0] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
  K_frag[1] = static_cast<uint32_t>(val >> 32);
}

// load Q from HBM to register. Permute columns
__device__ void load_Q_frag_permute_col(volatile uint32_t *Q_frag, uint64_t *Q, int embedding_dim, int rowIdx, int colIdx) {
    int laneid = threadIdx.x;//TODO: remove this
    int wid = threadIdx.y; //TODO: remove this
    uint64_t val = Q[rowIdx * embedding_dim + colIdx];
    Q_frag[0] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
    Q_frag[2] = static_cast<uint32_t>(val >> 32);
    val = Q[(rowIdx+8) * embedding_dim + colIdx];
    Q_frag[1] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
    Q_frag[3] = static_cast<uint32_t>(val >> 32);
}

// Assume Q is stored in row-major order.
//dyn_shm stores Q in 8x16 blocks where each block is stored in row-major order and blocks are stored in row-major order.
__device__ void load_Q_hbm2shm_128b(ulonglong2* __restrict__ Q_shm, ulonglong2* __restrict__ Q, int embedding_dim, int ind){
  ulonglong2 val = Q[ind];
  // /8 because ulonglong2 is 8 halfs
  int colid = ind % (embedding_dim/8);
  int block_colid = colid / 2;
  int rowid = ind / (embedding_dim/8);
  int block_offset = block_colid * BLK_M * 2 + rowid * 2 + colid % 2;
  Q_shm[block_offset] = val;
}

// Pair with load_Q_hbm2shm_128b. 
// This function has each warp read 1 16x16 block. 
//Not following the register layout in ptx doc but reordering it to match how K is loaded.
__device__ void load_Q_frag_shm(volatile uint32_t *Q_frag, uint64_t* __restrict__ dyn_shm, int ind, int laneid) {
  int offset = ind * BLK_M * BLK_M/4 + laneid;
  uint64_t val = dyn_shm[offset];
  Q_frag[0] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
  Q_frag[2] = static_cast<uint32_t>(val >> 32);
  val = dyn_shm[offset + 32];
  Q_frag[1] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
  Q_frag[3] = static_cast<uint32_t>(val >> 32);
}

// Each warp computes 1 tcb of S.
__global__ void sddmm_kernel_1tb1rw(
    const int *__restrict__ RowWindowOffset,
    const int *__restrict__ TCblockRowid,
    const int *__restrict__ SparseAToXidx, 
    const uint64_t *__restrict__ TCblockBitMap,
    int embeddingDim,
    torch::Half *__restrict__ Q, 
    torch::Half *__restrict__ K, 
    float2 *output) {
  volatile int laneid = threadIdx.x;
  int wid = threadIdx.y;
  volatile int tidInGroup = laneid % 4;
  // contains a RW of Q
  extern __shared__ uint32_t dyn_shm_1tb1rw[];
  
  for(int i = tid; i < BLK_M*embeddingDim/8; i += blockDim.x*blockDim.y){
    load_Q_hbm2shm_128b(reinterpret_cast<ulonglong2*>(dyn_shm_1tb1rw), reinterpret_cast<ulonglong2*>(Q+bid*BLK_M*embeddingDim), embeddingDim, i);
  }
  __syncthreads();

  for(int tcb_id = wid + RowWindowOffset[bid]; tcb_id < RowWindowOffset[bid+1]; tcb_id += blockDim.y) {
      volatile float S_frag[4] = {0.0f};
      volatile uint32_t Q_frag[4];
      volatile uint32_t K_frag[2];
      // int rowIdx_Q = TCblockRowid[tcb_id]*BLK_M + laneid/4;
      volatile int rowIdx_K = SparseAToXidx[tcb_id*BLK_N + laneid/4] * embeddingDim/4;
      for(int i = 0; i < embeddingDim/BLK_K; i++) {
        // load_Q_frag_permute_col(Q_frag, reinterpret_cast<uint64_t*>(Q), embeddingDim/4, rowIdx_Q, i*BLK_K/4 + tidInGroup);
        load_Q_frag_shm(Q_frag, reinterpret_cast<uint64_t*>(dyn_shm_1tb1rw), i, laneid);
        //load K with permuted columns
        uint64_t val = reinterpret_cast<uint64_t*>(K)[rowIdx_K + i*BLK_K/4 + tidInGroup];
        K_frag[0] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
        K_frag[1] = static_cast<uint32_t>(val >> 32);
        //load K unpermuted (will tank performance)
        // int rowIdx_K = SparseAToXidx[tcb_id*BLK_N + laneid/4] * embeddingDim/2; 
        // K_frag[0] = reinterpret_cast<uint32_t*>(K)[rowIdx_K + i*BLK_K/2 + tidInGroup];
        // K_frag[1] = reinterpret_cast<uint32_t*>(K)[rowIdx_K + i*BLK_K/2 + BLK_N/2 + tidInGroup];
        HMMA16816(S_frag[0], S_frag[1], S_frag[2], S_frag[3], 
                  Q_frag[0], Q_frag[1], Q_frag[2], Q_frag[3], 
                  K_frag[0], K_frag[1], 
                  S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
      }
      int bit_idx = 63 - laneid*2;
      for(int i = 0; i < 4; i++){
        uint64_t bit_mask = 1ULL << (bit_idx - i%2);
        S_frag[i] = (TCblockBitMap[tcb_id*2+i/2] & bit_mask) == 0 ? 0.0f : S_frag[i];
      }

      int offset = tcb_id * BLK_M * BLK_N;
      for(int j = 0; j < 2; j++){ // 2 8x8 blocks in each 16x8 block
        int sum_offset = j*BLK_N*BLK_N + laneid*2;
        float2 val;
        val.x = S_frag[j*2];
        val.y = S_frag[j*2+1];
        output[(offset + sum_offset)/2] = val;
      }

      //get the sum of D_frag among the threads in the warp
      // reduce_sum(D_frag, sum_max);
      //TODO: store sum to shared memory
      //get the max of D_frag among the threads in the warp
      // reduce_max(D_frag, sum_max);
      //TODO: store max to shared memory
      // __syncthreads();
  }
}

// Each tb computes multiple row windows of S
__global__ void sddmm_kernel_1tbnrw(
  const int *__restrict__ RowWindowOffset,
  const int *__restrict__ TBBoundaries,
  const int *__restrict__ TCblockRowid,
  const int *__restrict__ SparseAToXidx, 
  const uint64_t *__restrict__ TCblockBitMap,
  int embeddingDim,
  torch::Half *__restrict__ Q, 
  torch::Half *__restrict__ K, 
  float2 *output) {
    volatile int laneid = threadIdx.x;
    int wid = threadIdx.y;
    volatile int tidInGroup = laneid % 4;
    int tcb_start = RowWindowOffset[TBBoundaries[bid]];
    int tcb_end = RowWindowOffset[TBBoundaries[bid+1]];
    for(int tcb_id = tcb_start + wid; tcb_id < tcb_end; tcb_id += blockDim.y) {
        volatile float S_frag[4] = {0.0f};
        volatile uint32_t Q_frag[4];
        volatile uint32_t K_frag[2];
        int rowIdx_Q = TCblockRowid[tcb_id]*BLK_M + laneid/4;
        int rowIdx_K = SparseAToXidx[tcb_id*BLK_N + laneid/4] * embeddingDim/4; 
        for(int i = 0; i < embeddingDim/BLK_K; i++) {
            load_Q_frag_permute_col(Q_frag, reinterpret_cast<uint64_t*>(Q), embeddingDim/4, rowIdx_Q, i*BLK_K/4 + tidInGroup);
            uint64_t val = reinterpret_cast<uint64_t*>(K)[rowIdx_K + i*BLK_K/4 + tidInGroup];
            K_frag[0] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
            K_frag[1] = static_cast<uint32_t>(val >> 32);
            HMMA16816(S_frag[0], S_frag[1], S_frag[2], S_frag[3], 
                      Q_frag[0], Q_frag[1], Q_frag[2], Q_frag[3], 
                      K_frag[0], K_frag[1], 
                      S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
        }
        int bit_idx = 63 - laneid*2;
        for(int i = 0; i < 4; i++){
          uint64_t bit_mask = 1ULL << (bit_idx - i%2);
          S_frag[i] = (TCblockBitMap[tcb_id*2+i/2] & bit_mask) == 0 ? 0.0f : S_frag[i];
        }

        int offset = tcb_id * BLK_M * BLK_N;
        for(int j = 0; j < 2; j++){ // 2 8x8 blocks in each 16x8 block
          int sum_offset = j*BLK_N*BLK_N + laneid*2;
          float2 val;
          val.x = S_frag[j*2];
          val.y = S_frag[j*2+1];
          output[(offset + sum_offset)/2] = val;
        }

        //get the sum of D_frag among the threads in the warp
        // reduce_sum(D_frag, sum_max);
        //TODO: store sum to shared memory
        //get the max of D_frag among the threads in the warp
        // reduce_max(D_frag, sum_max);
        //TODO: store max to shared memory
        // __syncthreads();
    }
}
// #endif