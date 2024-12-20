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

// __device__ void set_Q_frag(uint32_t* Q_frag, half2* Q_half2, int bid, int wid, int laneid, int numNodes, int embedding_dim){
//    // Threads of a warp for fetching a 16X16 block of Q.
//   // DOC: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=wmma#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
//   // Here I'm swapping columns of Q to make the memory access more coalesced. 
//   // So when loading K, we have to swap the rows accordingly in order to get the same result.
//   int rowIdx = bid * BLK_M + laneid/4;
//   // /2 because half2. *2 because reading 2 consecutive half2. 
//   int colIdx = wid * BLK_K/2 + (laneid%4) * 2;
//   half2_uint32 h2U32Converter;
//   if(rowIdx < numNodes){
//     h2U32Converter.h2 = Q_half2[rowIdx * embedding_dim/2 + colIdx];
//     Q_frag[0] = h2U32Converter.u32;
//     h2U32Converter.h2 = Q_half2[rowIdx * embedding_dim/2 + colIdx + 1];
//     Q_frag[2] = h2U32Converter.u32;
//   }
//   if(rowIdx + 8 < numNodes){
//     h2U32Converter.h2 = Q_half2[(rowIdx+8) * embedding_dim/2 + colIdx];
//     Q_frag[1] = h2U32Converter.u32;
//     h2U32Converter.h2 = Q_half2[(rowIdx+8) * embedding_dim/2 + colIdx + 1];
//     Q_frag[3] = h2U32Converter.u32;
//   }
// }

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
__global__ void sddmm_kernel(
  int *RowWindowOffset,
  int *tcb_row_index,
  int *sparseAtoX, 
  const uint64_t *__restrict__ TCblock_bit_map,
  int embedding_dim,
  // int *TB_offset, 
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
  int nBlockEmbeddingDim = (embedding_dim + BLK_N - 1) / BLK_N;
  int nWarpPerBlock = (nBlockEmbeddingDim + TCBLOCK_PER_WARP - 1) / TCBLOCK_PER_WARP;
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

std::vector<torch::Tensor> f3S_sddmm_cuda(
    torch::Tensor Rowwindow_offset,
    torch::Tensor TCblock_rowid,
    torch::Tensor sparse_AToX_idx,
    torch::Tensor TCblock_bit_map,
    int num_nodes,
    int embedding_dim,
    torch::Tensor Q, torch::Tensor K){
  int nRowWindow = Rowwindow_offset.size(0) - 1;
  int nWarpPerBlock = 20;
  int nTCBlock = sparse_AToX_idx.size(0)/BLK_N;
  torch::Tensor sddmm_result = torch::zeros({nTCBlock*BLK_M*BLK_N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));  
  dim3 grid(nRowWindow, 1, 1);
  dim3 block(WARP_SIZE, nWarpPerBlock, 1);
  sddmm_kernel<<<grid, block>>>(
    Rowwindow_offset.data_ptr<int>(), 
    TCblock_rowid.data_ptr<int>(),
    sparse_AToX_idx.data_ptr<int>(),
    TCblock_bit_map.data_ptr<uint64_t>(),
    embedding_dim,
    Q.data_ptr<torch::Half>(), 
    K.data_ptr<torch::Half>(), 
    reinterpret_cast<float2*>(sddmm_result.data_ptr<float>()));
   // check for error
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
  return {sddmm_result};
}
// std::vector<torch::Tensor> fusedMM_forward_cuda(
// 	torch::Tensor Rowwindow_offset,
// 	torch::Tensor TCblocktile_id,
// 	torch::Tensor TCblock_offset,
// 	torch::Tensor sparse_AToX_idx,
// 	int num_nodes, int num_edges,
// 	int embedding_dim,  // embedding dimension.
// 	torch::Tensor input, // input feature matrix.
// 	bool save_edge_attention,
// 	bool use_f32_edge_attention,
// 	// default to m16n16k16
// 	bool use_m8n32k16 ) {
//   // warps per block
//   const int num_row_windows = Rowwindow_offset.size(0) - 1;
// 	int row_window_height;
// 	int nWarpPerBlock;
// 	if(use_m8n32k16){
// 		row_window_height = BLK_M;
// 		// Assuming embedding_dim is a multiple of 2*BLK_K
// 		nWarpPerBlock = (embedding_dim + BLK_K - 1) / BLK_K / TCBLOCK_PER_WARP_FMM;
// 	} else {
// 		row_window_height = BLK_H;
// 		nWarpPerBlock = (embedding_dim + row_window_height - 1) / row_window_height;
// 	}
// 	int paddedLength = num_row_windows * row_window_height;
//   auto output = torch::zeros({paddedLength, embedding_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
//   dim3 grid(num_row_windows, 1, 1);
//   dim3 block(WARP_SIZE, nWarpPerBlock, 1);
// 	torch::Tensor edgeAttention;
// 	if(use_f32_edge_attention) {
// 		edgeAttention = torch::zeros_like(TCblocktile_id).to(torch::kFloat32);
// 		if(use_m8n32k16){
//       #if BLK_M == 8 && BLK_N == 32 && BLK_K == 16
// 			int dynamic_shared_size = nWarpPerBlock * (BLK_M * BLK_N * sizeof(float) + BLK_N * BLK_N * sizeof(half));
// 			TC_fusedMM_fp32_inter_m8n32k16_cuda_kernel<<<grid, block, dynamic_shared_size>>>(
// 			Rowwindow_offset.data_ptr<int>(), 
// 			TCblocktile_id.data_ptr<uint8_t>(),
// 			TCblock_offset.data_ptr<int>(), 
// 			sparse_AToX_idx.data_ptr<int>(),
// 			num_nodes, num_edges, embedding_dim,
// 			input.data_ptr<torch::Half>(), 
// 			output.data_ptr<float>(),
// 			edgeAttention.data_ptr<float>(), 
// 			save_edge_attention);
//       #else
//       printf("m8n32k16 is not supported\n");
//       #endif
// 		}
// 		else{
// 			int dynamic_shared_size = nWarpPerBlock * BLK_H * BLK_H * (sizeof(half) + sizeof(float));
// 			TC_fusedMM_fp32_inter_cuda_kernel<<<grid, block, dynamic_shared_size>>>(
// 				Rowwindow_offset.data_ptr<int>(), 
// 				TCblocktile_id.data_ptr<uint8_t>(),
// 				TCblock_offset.data_ptr<int>(), 
// 				sparse_AToX_idx.data_ptr<int>(),
// 				num_nodes, num_edges, embedding_dim,
// 				input.data_ptr<torch::Half>(), 
// 				output.data_ptr<float>(),
// 				edgeAttention.data_ptr<float>(), 
// 				save_edge_attention);
// 		}
// 	}
// 	else{
// 		edgeAttention = torch::zeros_like(TCblocktile_id).to(torch::kHalf);
// 		const int dynamic_shared_size = 3 * nWarpPerBlock * BLK_H * BLK_H * sizeof(half);
// 		TC_fusedMM_cuda_kernel<<<grid, block, dynamic_shared_size>>>(
// 			Rowwindow_offset.data_ptr<int>(), 
// 			TCblocktile_id.data_ptr<uint8_t>(),
// 			TCblock_offset.data_ptr<int>(), 
// 			sparse_AToX_idx.data_ptr<int>(),
// 			num_nodes, num_edges, embedding_dim,
// 			input.data_ptr<float>(), 
// 			output.data_ptr<float>(),
// 			edgeAttention.data_ptr<torch::Half>(), 
// 			save_edge_attention);
// 	}
//   // check for error
//   cudaError_t error = cudaGetLastError();
//   if (error != cudaSuccess) {
//     // print the CUDA error message and exit
//     printf("CUDA error: %s\n", cudaGetErrorString(error));
//     exit(-1);
//   }
//   cudaDeviceSynchronize();
//   // remove padding
//   output = output.index(
//       {torch::indexing::Slice(0, num_nodes), torch::indexing::Slice()});
//   return {output, edgeAttention};
// }

// //////////////////////
// /// fusedMM
// /// should be launched with (embedding_dim + 16 - 1) / 16 warps of 32 threads
// /// note here we are assuming only 1 attention head
// //////////////////////
// __global__ void TC_fusedMM_cuda_kernel(
// 		const int *__restrict__ TCblock_rowid, 		 // offset of each row window.
// 		const uint8_t *__restrict__ TCblocktile_id,  // id of each TC block nonzero element.
// 		const int *__restrict__ TCblock_offset,      // colid of each TC block nonzero element.
// 		const int *__restrict__ sparse_AToX_idx,     // colid of each TC block nonzero element.
//     const int numNodes, const int numEdges,
//     const int embedding_dim,    // embedding dimension.
//     float *__restrict__ in_mat, // input feature matrix.
//     float *output,              // aggreAGNNed output feature matrix.
//     torch::Half *edgeAttention, // result of SDDMM.
//     bool save_edge_attention) {
//   int bid = blockIdx.x;     // block_index == row_window_index
//   int wid = threadIdx.y;    // warp_index handling multi-dimension > 16.
//   int laneid = threadIdx.x; // lanid of each warp.
//   int tid = wid * blockDim.x + laneid; // threadid of each block.

//   int warpPerBlock = blockDim.y;
//   int threadPerBlock = blockDim.x * warpPerBlock;

//   // starting node_id of current row_window.
//   int nid_start = bid * BLK_H; 
//   // ending node_id of the current row_window.
//   int nid_end = min((bid + 1) * BLK_H, numNodes); 
//   assert(nid_start < nid_end);

//   int warp_offset = wid * BLK_H * BLK_H;
//   __shared__ half edge_attention_block[BLK_H * BLK_H]; // Result of SDDMM

//   // each warp uses 3x16x16 shared memory
//   extern __shared__ half dynamic_shared[]; // 3 x blockDim.y x 16 x 16.
//   half *sparse_A_val = dynamic_shared;     // result of XX^T for all warps
//   half *dense_X_lst =
//       dynamic_shared + warpPerBlock * BLK_H * BLK_H; // X for all warps
//   half *dense_Y_lst =
//       dynamic_shared + 2 * warpPerBlock * BLK_H * BLK_H; // X^T for all warps

//   wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_H, half, wmma::row_major>
//       a_frag;
//   wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_H, half, wmma::col_major>
//       b_frag;
//   wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_H, half, wmma::row_major>
//       b_t_frag;
//   wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_H, half> acc_frag;
//   wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_H, float> spmm_acc_frag;
//   wmma::fill_fragment(spmm_acc_frag, 0.0f);

//   // Processing TC_blocks along the column dimension of Sparse A.
//   ///////////////////////////////////////////////////////////////
//   // Initialize dense_X by row-major store,
//   // Threads of a warp for fetching a dense_X.
//   // TODO: this is not coalesced. Make every 2 warps fetch their dense_X together to coalesce.
// #pragma unroll
//   for (int i = laneid; i < BLK_H * BLK_H; i += blockDim.x) {
//     int dense_rowIdx = i / BLK_H;
//     int dense_dimIdx = i % BLK_H;
//     int source_idx =
//         (nid_start + dense_rowIdx) * embedding_dim + wid * BLK_H + dense_dimIdx;
//     if (source_idx >= numNodes * embedding_dim)
//       dense_X_lst[i + warp_offset] = __float2half(0.0f);
//     else
//       dense_X_lst[i + warp_offset] = __float2half(in_mat[source_idx]);
//   }

//   /////////////////////////////////
//   // main loop
//   /////////////////////////////////
//   for (int tcb_id = TCblock_rowid[bid]; tcb_id < TCblock_rowid[bid+1]; tcb_id++) {
// 	// TODO: is this necessary? Feels like the one on line 5472 is enough
//     __syncthreads();

// #pragma unroll
//     for (int idx = tid; idx < BLK_H * BLK_H; idx += threadPerBlock) {
//       edge_attention_block[idx] = __float2half(0.0f);
//     }

// // Initialize dense_Y by column-major store,
// // Threads of a warp for fetching a dense_Y.
// // TODO: this is also not coalesced
// #pragma unroll
//     for (int i = laneid; i < BLK_H * BLK_H; i += blockDim.x) {
// 	  // TC block col ind to dense X row ind
//       int dense_rowIdx = sparse_AToX_idx[tcb_id * BLK_H + i / BLK_H]; 
//       // embedding_dim index of the dense tile.
//       int dense_dimIdx = i % BLK_H;    
//       int source_idx =
//           dense_rowIdx * embedding_dim + wid * BLK_H + dense_dimIdx;
//       if (source_idx >= numNodes * embedding_dim) {
//         dense_Y_lst[i + warp_offset] = __float2half(0.0f);
//       } else {
//         dense_Y_lst[i + warp_offset] = __float2half(in_mat[source_idx]);
//       }
//     }

//     wmma::load_matrix_sync(a_frag, dense_X_lst + warp_offset, BLK_H);
//     wmma::load_matrix_sync(b_frag, dense_Y_lst + warp_offset, BLK_H);
//     // clear acc_frag
//     wmma::fill_fragment(acc_frag, __float2half(0.0f));
//     // Perform the matrix multiplication on Tensor Core
//     wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
//     // Each warp store the result into the right slice of the intermediate
//     // tensor
//     wmma::store_matrix_sync(sparse_A_val + warp_offset, acc_frag, BLK_H,
//                             wmma::mem_row_major);

//     __syncthreads();

// 		// Starting and ending ind of nnz of the current TCblock in TCblocktile_id
// 		int eIdx_start = TCblock_offset[tcb_id];
// 		int nnz_in_block = TCblock_offset[tcb_id + 1] - eIdx_start;
// 			// Add up results from all warps using cuda cores
// 		// NOTE: if threadPerBlock > nnz_in_block, we only need 1 iteration. 
// 		// But if threadPerBlock < nnz_in_block, we need multiple iterations, which could slow things down.
// 		for(int i = tid; i < nnz_in_block; i += threadPerBlock) {
// 			int eid = eIdx_start + i;
// 			int block_id = TCblocktile_id[eid];
// 			for (int j = 0; j < warpPerBlock; j++) {
// 				// TODO: this access pattern doesn't seem very efficient
// 				edge_attention_block[block_id] =
// 					__hadd(edge_attention_block[block_id],
// 							sparse_A_val[block_id + j * BLK_H * BLK_H]);
// 			}
// 			// Save the edge attention
// 			// TODO: maybe we can coalesce this write
// 			if(save_edge_attention) {
// 				edgeAttention[eid] = edge_attention_block[block_id];
// 			}
// 		}
//     // necessary to ensure edge_attention_block is correct
//     __syncthreads();

//     /////////
//     // SpMM
//     /////////
//     // load the result of SDDMM
//     wmma::load_matrix_sync(a_frag, edge_attention_block, BLK_H);
//     // load feature matrix block
//     wmma::load_matrix_sync(b_t_frag, dense_Y_lst + warp_offset, BLK_H);

//     // spmm_acc_frag might be moved in and out of registers to local(global)
//     // memory every iteration, which can be bad (Ampere have 255 registers (32
//     // bits each) per thread) so we have to monitor for this, which I'm not sure
//     // how alternative is to compute and store the SDDMM results for all tc
//     // blocks first this will need (blockDim.y * 16 * 16 * 4) bytes
//     wmma::mma_sync(spmm_acc_frag, a_frag, b_t_frag, spmm_acc_frag);
//   }
//   wmma::store_matrix_sync(output + bid * BLK_H * embedding_dim + wid * BLK_H,
//                           spmm_acc_frag, embedding_dim, wmma::mem_row_major);
// }


// //////////////////////
// /// Same as fusedMM
// /// Except the partial result for SDDMM is stored in fp32 instead of half
// /// note here we are assuming only 1 attention head
// //////////////////////
// __global__ void TC_fusedMM_fp32_inter_cuda_kernel(
// 		const int *__restrict__ TCblock_rowid, 		 // offset of each row window.
// 		const uint8_t *__restrict__ TCblocktile_id,  // id of each TC block nonzero element.
// 		const int *__restrict__ TCblock_offset,      // colid of each TC block nonzero element.
// 		const int *__restrict__ sparse_AToX_idx,     // colid of each TC block nonzero element.
// 		const int numNodes, const int numEdges,
// 		const int embedding_dim,    // embedding dimension.
// 		torch::Half *__restrict__ in_mat, // input feature matrix.
// 		float *output,              // aggreAGNNed output feature matrix.
// 		float *edgeAttention, // result of SDDMM.
// 		bool save_edge_attention) {
//   int bid = blockIdx.x;     // block_index == row_window_index
//   int wid = threadIdx.y;    // warp_index handling multi-dimension > 16.
//   int laneid = threadIdx.x; // lanid of each warp.
//   int tid = wid * blockDim.x + laneid; // threadid of each block.

//   int warpPerBlock = blockDim.y;
//   int threadPerBlock = blockDim.x * warpPerBlock;

//   // starting node_id of current row_window.
//   int nid_start = bid * BLK_H; 
//   // ending node_id of the current row_window.
//   int nid_end = min((bid + 1) * BLK_H, numNodes); 
//   assert(nid_start < nid_end);

//   int warp_offset = wid * BLK_H * BLK_H;
//   __shared__ half edge_attention_block_half[BLK_H * BLK_H]; // Result of SDDMM
// 	__shared__ float edge_attention_block_single[BLK_H * BLK_H]; // Result of SDDMM in single

//   // each warp uses 2x16x16 shared memory
//   extern __shared__ char dynamic_shared_mixed[]; // 2 x blockDim.y x 16 x 16.
//   float* sparse_A_val = (float*) &dynamic_shared_mixed[0];     // result of XX^T for all warps
//   half* dense_X_lst =
//       (half*) &dynamic_shared_mixed[warpPerBlock * BLK_H * BLK_H * sizeof(float)]; // X for all warps

//   wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_H, half, wmma::row_major>
//       a_frag;
// 	wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_H, half, wmma::row_major>
//       att_frag;
//   wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_H, half, wmma::col_major>
//       b_frag;
//   wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_H, half, wmma::row_major>
//       b_t_frag;
//   wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_H, float> acc_frag;
//   wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_H, float> spmm_acc_frag;
//   wmma::fill_fragment(spmm_acc_frag, 0.0f);

//   // Processing TC_blocks along the column dimension of Sparse A.
//   ///////////////////////////////////////////////////////////////
//   // Initialize dense_X by row-major store,
//   // Threads of a warp for fetching a dense_X.
//   // TODO: this is not coalesced. Make every 2 warps fetch their dense_X together to coalesce.
// #pragma unroll
//   for (int i = laneid; i < BLK_H * BLK_H; i += blockDim.x) {
//     int dense_rowIdx = i / BLK_H;
//     int dense_dimIdx = i % BLK_H;
//     int source_idx =
//         (nid_start + dense_rowIdx) * embedding_dim + wid * BLK_H + dense_dimIdx;
//     if (source_idx >= numNodes * embedding_dim)
// 	  // TODO: is this conversion necessary?
//       dense_X_lst[i + warp_offset] = __float2half(0.0f);
//     else
//       dense_X_lst[i + warp_offset] = in_mat[source_idx];
//   }
// 	wmma::load_matrix_sync(a_frag, dense_X_lst + warp_offset, BLK_H);

// 	int tcb_id_start = TCblock_rowid[bid];
// 	int tcb_id_end = TCblock_rowid[bid + 1];
// 	// for loading dense_Y
// 	int n_col_group = BLK_H / (WARP_SIZE / BLK_H);
// 	int row = tid % embedding_dim;
// 	int block_row_id = row % BLK_H;
// 	int warp_start = (row / BLK_H) * BLK_H * BLK_H;
//   /////////////////////////////////
//   // main loop
//   /////////////////////////////////
//   for (int tcb_id = tcb_id_start; tcb_id < tcb_id_end; tcb_id++) {
// 		#pragma unroll
//     for (int idx = tid; idx < BLK_H * BLK_H; idx += threadPerBlock) {
//       edge_attention_block_half[idx] = __float2half(0.0f);
// 			edge_attention_block_single[idx] = 0.0f;
//     }

// 		// Initialize dense_Y by column-major store
// 		// Here I'm assuming embedding_dim is a multiple of BLK_H
// 		// and that each warp loads a BLK_H x BLK_H block
// 		// #pragma unroll
// 		for(int col_group = 0; col_group < n_col_group; col_group++){
// 			int block_col_id = col_group * 2 + (tid / embedding_dim);
// 			int X_rowId = sparse_AToX_idx[tcb_id * BLK_H + block_col_id];   
// 			dense_X_lst[block_col_id * BLK_H + block_row_id + warp_start] = in_mat[X_rowId * embedding_dim + row];
// 		}
// 		//also necessary
// 		__syncthreads();

//     wmma::load_matrix_sync(b_frag, dense_X_lst + warp_offset, BLK_H);
//     // clear acc_frag
//     wmma::fill_fragment(acc_frag, 0.0f);
//     // Perform the matrix multiplication on Tensor Core
//     wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
//     // Each warp store the result into the right slice of the intermediate
//     // tensor
//     wmma::store_matrix_sync(sparse_A_val + warp_offset, acc_frag, BLK_H,
//                             wmma::mem_row_major);

//     __syncthreads();

// 		// Starting and ending ind of nnz of the current TCblock in TCblocktile_id
// 		int eIdx_start = TCblock_offset[tcb_id];
// 		int nnz_in_block = TCblock_offset[tcb_id + 1] - eIdx_start;
// 			// Add up results from all warps using cuda cores
// 		// NOTE: if threadPerBlock > nnz_in_block, we only need 1 iteration. 
// 		// But if threadPerBlock < nnz_in_block, we need multiple iterations, which could slow things down.
// 		for(int i = tid; i < nnz_in_block; i += threadPerBlock) {
// 			int eid = eIdx_start + i;
// 			int block_id = TCblocktile_id[eid];
// 			for(int j = 0; j < warpPerBlock; j++) {
// 				// TODO: this access pattern doesn't seem very efficient
// 				edge_attention_block_single[block_id] += sparse_A_val[block_id + j * BLK_H * BLK_H];
// 			}
// 			edge_attention_block_half[block_id] = __float2half(edge_attention_block_single[block_id]);
// 			// Save the edge attention
// 			// TODO: maybe we can coalesce this write
// 			if(save_edge_attention) {
// 				edgeAttention[eid] = edge_attention_block_single[block_id];
// 			}
// 		}
//     // necessary to ensure edge_attention_block is correct
//     __syncthreads();

//     /////////
//     // SpMM
//     /////////
//     // load the result of SDDMM
//     wmma::load_matrix_sync(att_frag, edge_attention_block_half, BLK_H);
//     // load feature matrix block
//     wmma::load_matrix_sync(b_t_frag, dense_X_lst + warp_offset, BLK_H);

//     // spmm_acc_frag might be moved in and out of registers to local(global)
//     // memory every iteration, which can be bad (Ampere have 255 registers (32
//     // bits each) per thread) so we have to monitor for this, which I'm not sure
//     // how alternative is to compute and store the SDDMM results for all tc
//     // blocks first this will need (blockDim.y * 16 * 16 * 4) bytes
//     wmma::mma_sync(spmm_acc_frag, att_frag, b_t_frag, spmm_acc_frag);
//   }
//   wmma::store_matrix_sync(output + bid * BLK_H * embedding_dim + wid * BLK_H,
//                           spmm_acc_frag, embedding_dim, wmma::mem_row_major);
// }

// #if defined(BLK_M) && defined(BLK_N) && defined(BLK_K) && \
//     BLK_M == 8 && BLK_N == 32 && BLK_K == 16
// //////////////////////
// /// Same as fusedMM
// /// Except the partial result for SDDMM is stored in fp32 instead of half
// /// note here we are assuming only 1 attention head
// //////////////////////
// __global__ void TC_fusedMM_fp32_inter_m8n32k16_cuda_kernel(
// 		const int *__restrict__ TCblock_rowid, 		 // offset of each row window.
// 		const uint8_t *__restrict__ TCblocktile_id,  // id of each TC block nonzero element.
// 		const int *__restrict__ TCblock_offset,      // colid of each TC block nonzero element.
// 		const int *__restrict__ sparse_AToX_idx,     // colid of each TC block nonzero element.
// 		const int numNodes, const int numEdges,
// 		const int embedding_dim,    // embedding dimension.
// 		torch::Half *__restrict__ in_mat, // input feature matrix.
// 		float *output,              // aggreAGNNed output feature matrix.
// 		float *edgeAttention, // result of SDDMM.
// 		bool save_edge_attention) {
// 	half* in_mat_half = reinterpret_cast<half*>(in_mat);
//   int bid = blockIdx.x;     // block_index == row_window_index
//   int wid = threadIdx.y;    // warp_index handling multi-dimension > 16.
//   int laneid = threadIdx.x; // lanid of each warp.
//   int tid = wid * blockDim.x + laneid; // threadid of each block.

//   int warpPerBlock = blockDim.y;
//   int threadPerBlock = blockDim.x * warpPerBlock;

//   // starting node_id of current row_window.
//   int nid_start = bid * BLK_M; 
//   // ending node_id of the current row_window.
//   int nid_end = min((bid + 1) * BLK_M, numNodes); 
//   assert(nid_start < nid_end);

//   __shared__ half edge_attention_block_half[BLK_M * BLK_N]; // Result of SDDMM
// 	__shared__ float edge_attention_block_single[BLK_M * BLK_N]; // Result of SDDMM in single

//   extern __shared__ char dynamic_shared_mixed[]; 
//   float* sparse_A_val = (float*) &dynamic_shared_mixed[0];     // result of XX^T for all warps
//   // dense_X block for each warp. offset of wid*BLK_N*BLK_N is already included. 
//   half* dense_X =
//       (half*) &dynamic_shared_mixed[warpPerBlock * BLK_M * BLK_N * sizeof(float) + wid * BLK_N * BLK_N * sizeof(half)]; // X for all warps

//   wmma::fragment<wmma::matrix_a, BLK_M, BLK_N, BLK_K, half, wmma::row_major>
//       a_frag_0;
// 	wmma::fragment<wmma::matrix_a, BLK_M, BLK_N, BLK_K, half, wmma::row_major>
// 			a_frag_1;
// 	wmma::fragment<wmma::matrix_a, BLK_M, BLK_N, BLK_K, half, wmma::row_major>
// 			att_frag;
//   wmma::fragment<wmma::matrix_b, BLK_M, BLK_N, BLK_K, half, wmma::col_major>
//       b_frag;
//   wmma::fragment<wmma::matrix_b, BLK_M, BLK_N, BLK_K, half, wmma::row_major>
//       b_t_frag;
//   wmma::fragment<wmma::accumulator, BLK_M, BLK_N, BLK_K, float> acc_frag;
//   wmma::fragment<wmma::accumulator, BLK_M, BLK_N, BLK_K, float> spmm_acc_frag;
//   wmma::fill_fragment(spmm_acc_frag, 0.0f);

//   // load dense_X. each warp load 2 8x16 blocks.
// 	int warp_offset_X = bid * BLK_M * embedding_dim + wid * BLK_K * 2;
// 	wmma::load_matrix_sync(a_frag_0, in_mat_half + warp_offset_X, embedding_dim);
// 	wmma::load_matrix_sync(a_frag_1, in_mat_half + warp_offset_X + BLK_K, embedding_dim);

// 	int tcb_id_start = TCblock_rowid[bid];
// 	int tcb_id_end = TCblock_rowid[bid + 1];
//   /////////////////////////////////
//   // main loop
//   /////////////////////////////////
//   for (int tcb_id = tcb_id_start; tcb_id < tcb_id_end; tcb_id++) {
// 		#pragma unroll
//     for (int idx = tid; idx < BLK_M * BLK_N; idx += threadPerBlock) {
//       edge_attention_block_half[idx] = __float2half(0.0f);
// 			edge_attention_block_single[idx] = 0.0f;
//     }

// 		// Each warp loads a 32x32 block of compact(X^T) in column-major order
// 		for(int i = 0; i < BLK_N; i ++){
// 			int X_rowId = sparse_AToX_idx[tcb_id * BLK_N + i];
// 			dense_X[i * BLK_N + laneid] = in_mat_half[X_rowId * embedding_dim + wid * BLK_N + laneid];
// 		}
//     // print using tid == 0
//     if(tid == 0 && bid == 0 && tcb_id == 0){
//       for (int i = 0; i < 5; i++){
//         for (int j = 0; j < 5; j++){
//           printf("%f ", __half2float(dense_X[i * BLK_N + j]));
//         }
//         printf("\n");
//       }
//     }
// 		//also necessary
// 		__syncthreads();
// 		// __syncwarp();

//     int warp_offset_S = wid * BLK_M * BLK_N;
//     wmma::fill_fragment(acc_frag, 0.0f);
// 		for(int i = 0; i < 2; i++){
// 			wmma::load_matrix_sync(b_frag, dense_X + i*BLK_K, BLK_N);
// 			// Perform the matrix multiplication on Tensor Core
// 			if(i == 0)
// 				wmma::mma_sync(acc_frag, a_frag_0, b_frag, acc_frag);
// 			else
// 				wmma::mma_sync(acc_frag, a_frag_1, b_frag, acc_frag);
// 		}
//     // Each warp store the result into the right slice of the intermediate
//     // tensor
//     wmma::store_matrix_sync(sparse_A_val + warp_offset_S, acc_frag, BLK_N,
//                             wmma::mem_row_major);

//     __syncthreads();

//     if(tid == 0 && bid == 0 && tcb_id == 0){
//       int offset = warp_offset_S + 4*BLK_N;
//       printf("sparse_A_val partial warp 0:\n");
//       for (int i = 0; i < 25; i++){
//         printf("%f ", sparse_A_val[offset + i]);
//       }
//       printf("\n");

//       offset = warp_offset_S + 1*BLK_M*BLK_N + 4*BLK_N;
//       printf("sparse_A_val partial warp 1:\n");
//       for (int i = 0; i < 25; i++){
//         printf("%f ", sparse_A_val[offset + i]);
//       }
//       printf("\n");
//     }

// 		// Starting and ending ind of nnz of the current TCblock in TCblocktile_id
// 		int eIdx_start = TCblock_offset[tcb_id];
// 		int nnz_in_block = TCblock_offset[tcb_id + 1] - eIdx_start;
// 		// Add up results from all warps using cuda cores
// 		// NOTE: if threadPerBlock > nnz_in_block, we only need 1 iteration. 
// 		// But if threadPerBlock < nnz_in_block, we need multiple iterations, which could slow things down.
// 		for(int i = tid; i < nnz_in_block; i += threadPerBlock) {
// 			int eid = eIdx_start + i;
// 			int block_id = TCblocktile_id[eid];
// 			for(int j = 0; j < warpPerBlock; j++) {
// 				edge_attention_block_single[block_id] += sparse_A_val[block_id + j * BLK_M * BLK_N];
// 			}
// 			edge_attention_block_half[block_id] = __float2half(edge_attention_block_single[block_id]);
// 			// Save the edge attention
// 			if(save_edge_attention) {
// 				edgeAttention[eid] = edge_attention_block_single[block_id];
// 			}
// 		}
//     // necessary to ensure edge_attention_block is correct
//     __syncthreads();

//     /////////
//     // SpMM
//     /////////
// 		for(int i = 0; i < 2; i++){
// 			// load the result of SDDMM
// 			wmma::load_matrix_sync(att_frag, edge_attention_block_half + i*BLK_K, BLK_N);
// 			// load feature matrix block
// 			wmma::load_matrix_sync(b_t_frag, dense_X + i*BLK_K*BLK_N, BLK_N);
// 			// spmm_acc_frag might be moved in and out of registers to local(global)
// 			// memory every iteration, which can be bad (Ampere have 255 registers (32
// 			// bits each) per thread) so we have to monitor for this, which I'm not sure
// 			// how alternative is to compute and store the SDDMM results for all tc
// 			// blocks first this will need (blockDim.y * 16 * 16 * 4) bytes
// 			wmma::mma_sync(spmm_acc_frag, att_frag, b_t_frag, spmm_acc_frag);
// 		}
//   }
//   wmma::store_matrix_sync(output + bid * BLK_M * embedding_dim + wid * BLK_N,
//                           spmm_acc_frag, embedding_dim, wmma::mem_row_major);
// }
// #endif

#if defined(BLK_M) && defined(BLK_N) && defined(BLK_K) && \
    BLK_M == 16 && BLK_N == 8 && BLK_K == 16
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
 
  // 2 16x8 blocks, each block is divided into 2 8x8 subblocks in row major order.
  // __shared__ float sum[BLK_M * BLK_N * 2];
  // for(int i = tid; i < BLK_M * BLK_N * 2; i += blockDim.x * blockDim.y){
  //   sum[i] = 0.0f;
  // }
  float* sum = dyn_shm + BLK_M*2;
  // sddmm intermediate, 
  // TODO: this can probably be disallocated during spmm phase
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

__device__ void load_Q_frag_uint64(volatile uint32_t *Q_frag, uint64_t *Q, int embedding_dim, int rowIdx, int colIdx) {
    uint64_t val = Q[rowIdx * embedding_dim + colIdx];
    Q_frag[0] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
    Q_frag[2] = static_cast<uint32_t>(val >> 32);
    val = Q[(rowIdx+8) * embedding_dim + colIdx];
    Q_frag[1] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
    Q_frag[3] = static_cast<uint32_t>(val >> 32);
}

__global__ void sddmm_kernel(
  int *RowWindowOffset,
  int *tcb_row_index,
  int *sparseAtoX, 
  const uint64_t *__restrict__ TCblock_bit_map,
  int embedding_dim,
  // int *TB_offset, 
  torch::Half *__restrict__ Q, 
  torch::Half *__restrict__ K, 
  float2 *output) {
    volatile int laneid = threadIdx.x;
    int wid = threadIdx.y;
    volatile int tidInGroup = laneid % 4;
    for(int tcb_id = wid + RowWindowOffset[bid]; tcb_id < RowWindowOffset[bid+1]; tcb_id += blockDim.y) {
        volatile float S_frag[4] = {0.0f};
        volatile uint32_t Q_frag[4];
        volatile uint32_t K_frag[2];
        int rowIdx_Q = tcb_row_index[tcb_id]*BLK_M + laneid/4;
        int rowIdx_K = sparseAtoX[tcb_id*BLK_N + laneid/4] * embedding_dim/4; 
        for(int i = 0; i < embedding_dim/BLK_K; i++) {
            //load Q[row_start:row_start + BLK_M, i*BLK_N:i*BLK_N+BLK_N] using ldg64
            load_Q_frag_uint64(Q_frag, reinterpret_cast<uint64_t*>(Q), embedding_dim/4, rowIdx_Q, i*BLK_K/4 + tidInGroup);
            //load K
            uint64_t val = reinterpret_cast<uint64_t*>(K)[rowIdx_K + i*BLK_K/4 + tidInGroup];
            // if(bid==0){
            //   printf("laneid: %d, i: %d, rowIdx_Q: %d, colIdx_Q: %d rowIdx_K: %d, colIdx_K: %d\n", laneid, i, rowIdx_Q, i*BLK_K/4 + tidInGroup, rowIdx_K, i*BLK_K/4 + tidInGroup);
            //   printf("laneid: %d, Q_frag: %d %d %d %d\n", laneid, Q_frag[0], Q_frag[1], Q_frag[2], Q_frag[3]);
            //   printf("laneid: %d, val: %llu\n", laneid, val);
            // }
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
          S_frag[i] = (TCblock_bit_map[tcb_id*2+i/2] & bit_mask) == 0 ? 0.0f : S_frag[i];
        }

        // if(bid ==0 && wid == 1 && tcb_id == 1){
        //   printf("laneid: %d, Q_frag: %d %d %d %d\n", laneid, Q_frag[0], Q_frag[1], Q_frag[2], Q_frag[3]);
        //   printf("laneid: %d, K_frag: %d %d\n", laneid, K_frag[0], K_frag[1]);
        //   printf("laneid: %d, S_frag: %f %f %f %f\n", laneid, S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
        // }

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
#endif