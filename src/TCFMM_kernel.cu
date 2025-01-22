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

union Half2Uint32 {
    half2 h2;
    uint32_t u32;
};

struct Scheduler {
  int ind = blockIdx.x;
  int targetRw;

  __device__ bool 
  next_iter(const int* sortedRowWindows, int nRw){
    if(ind < nRw){
      targetRw = sortedRowWindows[ind];
      ind += gridDim.x;
      return true;
    }
    else{
      return false;
    }
  }
};

// Assume each warp has a 8x8 fp16 matrix in row-major order distributed among threads
// Each thread starts with 2 consecutive fp16 values as a half2 (val)
// This function redistributes elements among threads 
// so that val becomes 2 consecutive fp16 values in column-major order
__device__ void shflTransposeWarp(half2 &val, int laneId){
  int col = laneId/4;
  int row = (laneId%4)*2;
  half2 temp[2];
  temp[0] = __shfl_sync(0xffffffff, val, row*4 + col/2);
  temp[1] = __shfl_sync(0xffffffff, val, (row+1)*4 + col/2);
  if((laneId/4)%2 == 0){
    val.x = temp[0].x;
    val.y = temp[1].x;
  }
  else{
    val.x = temp[0].y;
    val.y = temp[1].y;
  }
}

__device__ void saveSddmmResult(float* sum, float* sddmm_result, int tcbId, int last_block){
  int warpId = threadIdx.y;    // warp_index handling multi-dimension > 16.
  int laneId = threadIdx.x; // lanid of each warp.
  // have warp 0 load sum into sddmm_result, which is block row major
  if(warpId == 0){
    int offset = tcbId * BLK_M * BLK_N;
    for(int i = 0; i < 2; i++){ // 2 16x8 blocks
      if(!last_block || i == 0){
        for(int j = 0; j < 2; j++){ // 2 8x8 blocks in each 16x8 block
          int sum_offset = i*BLK_M*BLK_N + j*BLK_N*BLK_N + laneId*2;
          sddmm_result[offset + sum_offset] = sum[sum_offset];
          sddmm_result[offset + sum_offset + 1] = sum[sum_offset + 1];
        }
      }
    }
  }
}

// produce the max value for every subwarp of 4 threads
__device__ void reduceMax(float& max, volatile int laneId){
  // Round 1: Compare with neighbor at offset 1
  max = fmaxf(max, __shfl_xor_sync(0xF << (laneId/4)*4, max, 1, 4));
  // Round 2: Compare with neighbor at offset 2
  max = fmaxf(max, __shfl_xor_sync(0xF << (laneId/4)*4, max, 2, 4));
}

// sum up "sum" for every 4 consecutive threads in a warp
// results is only valid for the first thread in the warp
__device__ void reduceSum(float& sum){
  // offset = 1
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 1, 4);
  // offset = 2
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 2, 4);
}

// deprecated, replaced by storePartialSumShm and addPartialSums
// use atomicAdd to sum up the result of SDDMM.
__device__ void sumWarpI(int i, const uint64_t* tcbBitMap, float* sum, float* D_frag, int tcbId, bool lastBlock, int laneId){
  uint64_t bitMask = 1ULL << (63 - laneId*2);
  uint64_t bitMaskNext = 1ULL << (63 - laneId*2-1);
  if(!lastBlock || i == 0){
    int sumOffset = i*BLK_M*BLK_N;
    for(int j=0; j< 2; j++){// 2 8x8 blocks in each 16x8 block
      if((tcbBitMap[(tcbId+i)*2+j] & bitMask) != 0){
        atomicAdd(&sum[sumOffset + j*BLK_N*BLK_N + laneId*2], D_frag[j*2]);
      }
      if((tcbBitMap[(tcbId+i)*2+j] & bitMaskNext) != 0){
        atomicAdd(&sum[sumOffset + j*BLK_N*BLK_N + laneId*2 + 1], D_frag[j*2 + 1]);
      }
    }
  }
}

//sum should be BLK_M*BLK_M*number of warps.
__device__ void storePartialSumShm(int i, float* sum, float* D_frag, int warpId, int laneId){
  int sumOffset = warpId*BLK_M*BLK_M + i*BLK_M*BLK_N + laneId*2;
  sum[sumOffset] = D_frag[0];
  sum[sumOffset + 1] = D_frag[1];
  sum[sumOffset + BLK_N*BLK_N] = D_frag[2];
  sum[sumOffset + BLK_N*BLK_N + 1] = D_frag[3];
}

__device__ void addPartialSums(float* sum, int tcbId, int tid, int n_warps, const uint64_t* TCblock_bit_map, bool last_block){
  for(int ind = tid; ind < BLK_M*BLK_M; ind += blockDim.x*blockDim.y){
    int block_id = ind / 64; // which 8x8 block ind belongs to
    if(!last_block || block_id < 2){
      int block_offset = ind % 64; // which element in the 8x8 block ind belongs to
      uint64_t bit_mask = 1ULL << (63 - block_offset);
      if((TCblock_bit_map[tcbId*2+block_id] & bit_mask) != 0){
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

__device__ void setQFrag(volatile uint32_t* Q_frag, uint64_t* Q, int bid, int warpId, int laneId, int numNodes, int embeddingDim){
   // Threads of a warp for fetching a 16X16 block of Q.
  // DOC: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=wmma#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
  // Here I'm swapping columns of Q to make the memory access more coalesced. 
  // So when loading K, we have to swap the rows accordingly in order to get the same result.
  int rowIdx = bid * BLK_M + laneId/4;
  // /4 because half4.
  int colIdx = warpId * BLK_K/4 + (laneId%4);
  if(rowIdx < numNodes){
    uint64_t val = Q[rowIdx * embeddingDim/4 + colIdx];
    Q_frag[0] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
    Q_frag[2] = static_cast<uint32_t>(val >> 32);
  }
  if(rowIdx + 8 < numNodes){
    uint64_t val = Q[(rowIdx+8) * embeddingDim/4 + colIdx];
    Q_frag[1] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
    Q_frag[3] = static_cast<uint32_t>(val >> 32);
  }
}

__device__ void setQFragShm(float* dynShm, uint64_t* Q, int bid, int warpId, int laneId, int numNodes, int embeddingDim){
  uint32_t* Q_frag = reinterpret_cast<uint32_t*>(dynShm + BLK_M*2 + warpId*blockDim.x*4);
  int rowIdx = bid * BLK_M + laneId/4;
  // /4 because half4.
  int colIdx = warpId * BLK_K/4 + (laneId%4);
  if(rowIdx < numNodes){
    uint64_t val = Q[rowIdx * embeddingDim/4 + colIdx];
    Q_frag[laneId] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
    Q_frag[blockDim.x*2 + laneId] = static_cast<uint32_t>(val >> 32);
  }
  if(rowIdx + 8 < numNodes){
    uint64_t val = Q[(rowIdx+8) * embeddingDim/4 + colIdx];
    Q_frag[blockDim.x + laneId] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
    Q_frag[blockDim.x*3 + laneId] = static_cast<uint32_t>(val >> 32);
  }
}

// __global__ void TC_fusedMM_cuda_kernel(
// 	const int *__restrict__ TCblock_rowid, 		 // offset of each row window.
// 	const uint8_t *__restrict__ TCblocktile_id,  // id of each TC block nonzero element.
// 	const int *__restrict__ TCblock_offset,      // colid of each TC block nonzero element.
// 	const int *__restrict__ sparse_AToX_idx,     // colid of each TC block nonzero element.
// 	const int numNodes, const int numEdges,
// 	const int embeddingDim,    // embedding dimension.
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
// 	const int embeddingDim,    // embedding dimension.
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
		const int embeddingDim,    // embedding dimension.
		torch::Half *__restrict__ in_mat, // input feature matrix.
		float *output,              // aggreAGNNed output feature matrix.
		float *edgeAttention, // result of SDDMM.
		bool save_edge_attention);
#endif

__global__ void f3sKernel1tb1tcb(
		const int *__restrict__ rowWindowOffset, 		 // offset of each row window.
		const int *__restrict__ sparseAToXidx,     // colid of each TC block nonzero element.
    const uint64_t *__restrict__ tcbBitMap,
    const int numNodes,
    const int embeddingDim,
		torch::Half *__restrict__ Q, 
    torch::Half *__restrict__ K, 
    torch::Half *__restrict__ V,
		float *output,              // output feature matrix.
		float *sddmmResult, // result of SDDMM
    bool applySoftmax
);

__global__ void f3sKernel1tb1rw(
    const int *__restrict__ rowWindowOffset,
    const int *__restrict__ sparseAToXidx, 
    const uint64_t *__restrict__ tcbBitMap,
    int embeddingDim,
    ulonglong2 *__restrict__ Q, 
    ulonglong2 *__restrict__ K,
    half *__restrict__ V,
    float2 *output,
    float2 *sddmmResult);

__global__ void f3sKernel1tb1rwScheduled(
    const int *__restrict__ rowWindowOffset,
    const int *__restrict__ sortedRowWindows,
    const int *__restrict__ sparseAToXidx, 
    const uint64_t *__restrict__ tcbBitMap,
    int embeddingDim,
    int nRw,
    ulonglong2 *__restrict__ Q, 
    ulonglong2 *__restrict__ K, 
    half *__restrict__ V,
    float2 *output,
    float2 *sddmmResult);

__global__ void f2sKernel1tb1rw(
    const int *__restrict__ rowWindowOffset,
    const int *__restrict__ sparseAToXidx, 
    const uint64_t *__restrict__ tcbBitMap,
    int embeddingDim,
    ulonglong2 *__restrict__ Q, 
    ulonglong2 *__restrict__ K, 
    half *__restrict__ V,
    float2 *output,
    float2 *sddmmResult);

__global__ void sddmmKernel1tbnrw(
  const int *__restrict__ rowWindowOffset,
  const int *__restrict__ tbBoundaries,
  const int *__restrict__ tcbRowid,
  const int *__restrict__ sparseAToXidx, 
  const uint64_t *__restrict__ tcbBitMap,
  int embeddingDim,
  torch::Half *__restrict__ Q, 
  torch::Half *__restrict__ K, 
  float2 *output);

std::vector<torch::Tensor> 
f3sCuda1tb1tcb(
    torch::Tensor rowWindowOffset,
    torch::Tensor sparseAToXidx, 
    torch::Tensor tcbBitMap,
    int numNodes, 
    int embeddingDim,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, 
    bool applySoftmax,
    bool saveSddmmResult){
  int nBlockEmbeddingDim = (embeddingDim + BLK_N - 1) / BLK_N;
  int nWarpPerBlock =  (nBlockEmbeddingDim + 2 - 1) / 2;
  const int nRowWindow = rowWindowOffset.size(0) - 1;
  int paddedLength = nRowWindow * BLK_M;
  auto output = torch::zeros({paddedLength, embeddingDim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  dim3 grid(nRowWindow, 1, 1);
  dim3 block(WARP_SIZE, nWarpPerBlock, 1);
  int nTcb = sparseAToXidx.size(0)/BLK_N;
	torch::Tensor sddmmResult = torch::zeros({nTcb*BLK_M*BLK_N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  float* sddmmResultPtr = saveSddmmResult ? sddmmResult.data_ptr<float>() : nullptr;
  int fixedSharedSize = nWarpPerBlock * BLK_M * BLK_M * sizeof(float);
  int dynamicSharedSize = applySoftmax ? fixedSharedSize + 2 * BLK_M * sizeof(float) : fixedSharedSize;
  #if BLK_M == 16 && BLK_N == 8 && BLK_K == 16
  f3sKernel1tb1tcb<<<grid, block, dynamicSharedSize>>>(
    rowWindowOffset.data_ptr<int>(), 
    sparseAToXidx.data_ptr<int>(),
    tcbBitMap.data_ptr<uint64_t>(),
    numNodes, embeddingDim,
    Q.data_ptr<torch::Half>(), 
    K.data_ptr<torch::Half>(), 
    V.data_ptr<torch::Half>(),
    output.data_ptr<float>(),
    sddmmResultPtr,
    applySoftmax);
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
      {torch::indexing::Slice(0, numNodes), torch::indexing::Slice()});
  return {output, sddmmResult};
}

std::vector<torch::Tensor> 
f3sCuda1tb1rw(
    torch::Tensor rowWindowOffset,
    torch::Tensor sparseAToXidx,
    torch::Tensor tcbBitMap,
    int nNodes,
    int embeddingDim,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int nWarpPerBlock,
    bool applySoftmax){
  int nTcb = sparseAToXidx.size(0)/BLK_M;
  torch::Tensor sddmmResult = torch::zeros({nTcb*BLK_M*BLK_M}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)); 
  int nRowWindow = rowWindowOffset.size(0) - 1;
  int paddedLength = nRowWindow * BLK_M; 
  auto output = torch::zeros({paddedLength, embeddingDim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  int sharedSize = BLK_M * embeddingDim * sizeof(half); // Q
  sharedSize += nWarpPerBlock * BLK_M * BLK_N * sizeof(half); // E
  sharedSize += nWarpPerBlock * 2 * BLK_M * sizeof(float); // row_max, row_sum, old_max, old_sum
  sharedSize += BLK_M * embeddingDim * sizeof(float); // O_frag
  printf("sharedSize: %d\n", sharedSize);
  dim3 grid(nRowWindow, 1, 1);
  dim3 block(WARP_SIZE, nWarpPerBlock, 1);
  if(applySoftmax){
    f3sKernel1tb1rw<<<grid, block, sharedSize>>>(
      rowWindowOffset.data_ptr<int>(), 
      sparseAToXidx.data_ptr<int>(),
      tcbBitMap.data_ptr<uint64_t>(),
      embeddingDim,
      reinterpret_cast<ulonglong2*>(Q.data_ptr<torch::Half>()), 
      reinterpret_cast<ulonglong2*>(K.data_ptr<torch::Half>()), 
      reinterpret_cast<half*>(V.data_ptr<torch::Half>()),
      reinterpret_cast<float2*>(output.data_ptr<float>()),
      reinterpret_cast<float2*>(sddmmResult.data_ptr<float>()));
  }
  else{
    f2sKernel1tb1rw<<<grid, block, sharedSize>>>(
      rowWindowOffset.data_ptr<int>(), 
      sparseAToXidx.data_ptr<int>(),
      tcbBitMap.data_ptr<uint64_t>(),
      embeddingDim,
      reinterpret_cast<ulonglong2*>(Q.data_ptr<torch::Half>()), 
      reinterpret_cast<ulonglong2*>(K.data_ptr<torch::Half>()), 
      reinterpret_cast<half*>(V.data_ptr<torch::Half>()),
      reinterpret_cast<float2*>(output.data_ptr<float>()),
      reinterpret_cast<float2*>(sddmmResult.data_ptr<float>()));
  }
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  // remove padding
  output = output.index(
      {torch::indexing::Slice(0, nNodes), torch::indexing::Slice()});
  return {output, sddmmResult};
}

std::vector<torch::Tensor> 
f3sCuda1tb1rwScheduled(
    torch::Tensor rowWindowOffset,
    torch::Tensor sortedRowWindows,
    torch::Tensor sparseAToXidx,
    torch::Tensor tcbBitMap,
    int nNodes,
    int embeddingDim,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int nWarpPerBlock){
  int nTcb = sparseAToXidx.size(0)/BLK_M;
  torch::Tensor sddmmResult = torch::zeros({nTcb*BLK_M*BLK_M}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)); 
  int nRowWindow = rowWindowOffset.size(0) - 1;
  int paddedLength = nRowWindow * BLK_M; 
  auto output = torch::zeros({paddedLength, embeddingDim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  int sharedSize = BLK_M * embeddingDim * sizeof(half); // Q
  sharedSize += nWarpPerBlock * BLK_M * BLK_N * sizeof(half); // E
  sharedSize += nWarpPerBlock * 2 * BLK_M * sizeof(float); // row_max, row_sum, old_max, old_sum
  sharedSize += BLK_M * embeddingDim * sizeof(float); // O_frag
  printf("sharedSize: %d\n", sharedSize);
  dim3 grid(nRowWindow, 1, 1);
  dim3 block(WARP_SIZE, nWarpPerBlock, 1);
  f3sKernel1tb1rwScheduled<<<grid, block, sharedSize>>>(
    rowWindowOffset.data_ptr<int>(), 
    sortedRowWindows.data_ptr<int>(), 
    sparseAToXidx.data_ptr<int>(),
    tcbBitMap.data_ptr<uint64_t>(),
    embeddingDim,
    nRowWindow,
    reinterpret_cast<ulonglong2*>(Q.data_ptr<torch::Half>()), 
    reinterpret_cast<ulonglong2*>(K.data_ptr<torch::Half>()), 
    reinterpret_cast<half*>(V.data_ptr<torch::Half>()),
    reinterpret_cast<float2*>(output.data_ptr<float>()),
    reinterpret_cast<float2*>(sddmmResult.data_ptr<float>()));
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  // remove padding
  output = output.index(
      {torch::indexing::Slice(0, nNodes), torch::indexing::Slice()});
  return {output, sddmmResult};
}

std::vector<torch::Tensor> 
sddmmCuda1tbnrw(
    torch::Tensor rowWindowOffset,
    torch::Tensor tbBoundaries,
    torch::Tensor tcbRowid,
    torch::Tensor sparseAToXidx,
    torch::Tensor tcbBitMap,
    int nNodes,
    int embeddingDim,
    torch::Tensor Q, torch::Tensor K,
    int nWarpPerBlock){
  int nRowWindow = tbBoundaries.size(0) - 1;
  int nTcb = sparseAToXidx.size(0)/BLK_N;
  torch::Tensor sddmmResult = torch::zeros({nTcb*BLK_M*BLK_N}, 
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));  
  dim3 grid(nRowWindow, 1, 1);
  dim3 block(WARP_SIZE, nWarpPerBlock, 1);
  sddmmKernel1tbnrw<<<grid, block>>>(
    rowWindowOffset.data_ptr<int>(), 
    tbBoundaries.data_ptr<int>(),
    tcbRowid.data_ptr<int>(),
    sparseAToXidx.data_ptr<int>(),
    tcbBitMap.data_ptr<uint64_t>(),
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
#define tid (threadIdx.x + threadIdx.y * blockDim.x)
__global__ void f3sKernel1tb1tcb(
		const int *__restrict__ rowWindowOffset, 		 // offset of each row window.
		const int *__restrict__ sparseAToXidx,     // colid of each TC block nonzero element.
    const uint64_t *__restrict__ tcbBitMap,
    const int numNodes,
    const int embeddingDim,
		torch::Half *__restrict__ Q, 
    torch::Half *__restrict__ K, 
    torch::Half *__restrict__ V,
		float *output,
		float *sddmmResult,
    bool applySoftmax){
  volatile int warpId = threadIdx.y;    // warp_index handling multi-dimension > 16.
  volatile int laneId = threadIdx.x; // lanid of each warp.

  uint64_t* K_uint64 = reinterpret_cast<uint64_t*>(K);

  // row_max, row_sum (size BLK_M each) for online-softmax,
  // then Q_frag, then O_frag
  extern __shared__ float dynShm[]; 
  for(int i = tid; i < BLK_M*2+blockDim.y*BLK_M*BLK_M; i += blockDim.x*blockDim.y){
  // for(int i = tid; i < BLK_M*2; i += blockDim.x*blockDim.y){
    dynShm[i] = 0.0f;
  }
 
  float* sum = dynShm + BLK_M*2;
  float O_frag[8] = {0};// spmm result
  uint32_t Q_frag[4] = {0};
  setQFrag(Q_frag, reinterpret_cast<uint64_t*>(Q), bid, warpId, laneId, numNodes, embeddingDim);
  // setQFragShm(dynShm, reinterpret_cast<uint64_t*>(Q), bid, warpId, laneId, numNodes, embeddingDim);

  /////////////////////////////////
  // main loop
  /////////////////////////////////
  volatile bool lastBlock = false;
  for (int tcbId = rowWindowOffset[bid]; tcbId < rowWindowOffset[bid + 1]; tcbId+=2) {
    if((rowWindowOffset[bid + 1] - rowWindowOffset[bid]) % 2 && tcbId == rowWindowOffset[bid + 1] - 1){
      lastBlock = true;
    }
    {// sddmm
      uint32_t B_frag[2];
      float D_frag[4];
      // uint32_t* Q_frag = reinterpret_cast<uint32_t*>(dynShm + BLK_M*2 + warpId*blockDim.x*4);
      int colIdx = (warpId * BLK_M)/4 + (laneId % 4); 
      for(int i = 0; i < 2; i++){
        if(!lastBlock || i == 0){
          // Initialize B_frag from K
          // Note I'm swapping rows of B_frag because we swapped the columns of A_frag(Q)
          // index in terms of half2, only affect rowIdx
          int rowIdx = sparseAToXidx[(tcbId+i) * BLK_N + laneId / 4]; 
          uint64_t val = K_uint64[rowIdx * embeddingDim/4 + colIdx];
          B_frag[0] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
          B_frag[1] = static_cast<uint32_t>(val >> 32);
          HMMA16816(D_frag[0], D_frag[1], D_frag[2], D_frag[3], 
                    // Q_frag[laneId], Q_frag[blockDim.x + laneId], Q_frag[blockDim.x*2 + laneId], Q_frag[blockDim.x*3 + laneId], 
                    Q_frag[0], Q_frag[1], Q_frag[2], Q_frag[3], 
                    B_frag[0], B_frag[1], 
                    0.0f, 0.0f, 0.0f, 0.0f);
          storePartialSumShm(i, sum, D_frag, warpId, laneId);
        }
      }
      __syncthreads();
      addPartialSums(sum, tcbId, tid, blockDim.y, tcbBitMap, lastBlock);
    }
    __syncthreads();

    if(sddmmResult != nullptr){
      saveSddmmResult(sum, sddmmResult, tcbId, lastBlock);
    }

    {// softmax + spmm
      uint32_t S_frag[4];// softmax/sddmm result
      if(applySoftmax){
        float D_frag[4];
        float2* sumFloat2 = reinterpret_cast<float2*>(sum);
        for(int j = 0; j < 2; j++){// 2 8x8 blocks in each 16x8 block
          int sumOffset = j*BLK_N*BLK_N/2;
          for(int i = 0; i < 2; i++){// 2 16x8 blocks
            if(!lastBlock || i == 0){
              float2 temp = sumFloat2[i*BLK_M*BLK_N/2 + sumOffset + laneId];
              D_frag[i*2] = temp.x;
              D_frag[i*2 + 1] = temp.y;
            }
            else{
              D_frag[i*2] = 0.0f;
              D_frag[i*2 + 1] = 0.0f;
            }
          }
          //max of the 4 elements in the same row across 2 16x8 blocks
          //need every warp to do this because they will need it for the next computation
          float maxOld = dynShm[j*BLK_N + laneId/4];

          float max = fmaxf(
            fmaxf(fmaxf(D_frag[0], D_frag[1]), fmaxf(D_frag[2], D_frag[3])), maxOld);
          reduceMax(max, laneId);

          for(int i = 0; i < 4; i++){
            if(D_frag[i] != 0.0f){
              D_frag[i] = __expf(D_frag[i] - max);
            }
          }

          float expMaxDiff = __expf(maxOld - max);

          if(warpId == 0){
            float sum = D_frag[0] + D_frag[1] + D_frag[2] + D_frag[3];
            reduceSum(sum);
            if(laneId % 4 == 0){
              dynShm[BLK_M + j*BLK_N + laneId/4] = dynShm[BLK_M + j*BLK_N + laneId/4] * expMaxDiff + sum;
            }
          }

          O_frag[j*2]   = O_frag[j*2]   * expMaxDiff;
          O_frag[j*2+1] = O_frag[j*2+1] * expMaxDiff;
          O_frag[j*2+4] = O_frag[j*2+4] * expMaxDiff;
          O_frag[j*2+5] = O_frag[j*2+5] * expMaxDiff;
          // float* O_frag = dynShm + BLK_M*2 + blockDim.x*blockDim.y*4 + warpId*blockDim.x*8;
          // O_frag[blockDim.x*j*2 + laneId]     = O_frag[blockDim.x*j*2 + laneId]   * (expMaxDiff);
          // O_frag[blockDim.x*(j*2+1) + laneId] = O_frag[blockDim.x*(j*2+1) + laneId] * (expMaxDiff);
          // O_frag[blockDim.x*(j*2+4) + laneId] = O_frag[blockDim.x*(j*2+4) + laneId] * (expMaxDiff);
          // O_frag[blockDim.x*(j*2+5) + laneId] = O_frag[blockDim.x*(j*2+5) + laneId] * (expMaxDiff);

          if(warpId == 0 && laneId % 4 == 0){
            dynShm[j*BLK_N + laneId/4] = max;
          }

          Half2Uint32 h2U32Converter;
          for(int i = 0; i < 2; i++){
            h2U32Converter.h2.x = __float2half(D_frag[i*2]);
            h2U32Converter.h2.y = __float2half(D_frag[i*2+1]);
            S_frag[i*2 + j] = h2U32Converter.u32;
          }
        }
      }
      else{
        float2* sumFloat2 = reinterpret_cast<float2*>(sum);
        for(int i = 0; i < 2; i++){// 2 16x8 blocks
          int sumOffset = i*BLK_M*BLK_N/2;
          Half2Uint32 h2U32Converter;
          if(!lastBlock || i == 0){
            for(int j = 0; j < 2; j++){// 2 8x8 blocks in each 16x8 block
              float2 temp = sumFloat2[sumOffset + j*BLK_N*BLK_N/2 + laneId];
              h2U32Converter.h2.x = __float2half(temp.x);
              h2U32Converter.h2.y = __float2half(temp.y);
              S_frag[i*2+j] = h2U32Converter.u32;
            }
          }
          else{
            S_frag[i*2] = 0;
            S_frag[i*2+1] = 0;
          }
        }
      }
      __syncthreads();
      //reset sum to 0
      // for(int i = tid; i < BLK_M * BLK_N * 2; i += blockDim.x * blockDim.y){
      //   sum[i] = 0.0f;
      // }
      for(int i = laneId; i < BLK_M*BLK_M; i += blockDim.x){
        sum[warpId*BLK_M*BLK_M + i] = 0.0f;
      }
      /////////
      // SpMM
      /////////
      {
        uint32_t B_frag[2];
        Half2Uint32 h2U32Converter;
        half temp_V[2];
        // float* O_frag = dynShm + BLK_M*2 + blockDim.x*blockDim.y*4 + warpId*blockDim.x*8;
        for(int j = 0; j < 2; j++){// 2 16x8 blocks
          int colIdx = (warpId*2+j) * BLK_N + laneId/4;
          for(int i = 0; i < 2; i++){// 2 8x8 blocks in each 16x8 block
            if(!lastBlock || i == 0){
              for(int k = 0; k < 2; k++){// 2 halfs in each 8x8 block
                int rowIdx = sparseAToXidx[(tcbId+i) * BLK_N + (laneId%4)*2 + k];
                temp_V[k] = V[rowIdx * embeddingDim + colIdx];
              }
              h2U32Converter.h2 = __halves2half2(temp_V[0], temp_V[1]);
              B_frag[i] = h2U32Converter.u32;
            }
            else{
              B_frag[i] = 0;
            }
          }
          // HMMA16816(O_frag[blockDim.x*j*4 + laneId], O_frag[blockDim.x*(j*4+1) + laneId], O_frag[blockDim.x*(j*4+2) + laneId], O_frag[blockDim.x*(j*4+3) + laneId], 
          HMMA16816(O_frag[4*j], O_frag[4*j+1], O_frag[4*j+2], O_frag[4*j+3],
                    S_frag[0], S_frag[1], S_frag[2], S_frag[3], 
                    B_frag[0], B_frag[1], 
                 // O_frag[blockDim.x*4*j + laneId], O_frag[blockDim.x*(4*j+1) + laneId], O_frag[blockDim.x*(4*j+2) + laneId], O_frag[blockDim.x*(4*j+3) + laneId]);
                    O_frag[4*j], O_frag[4*j+1], O_frag[4*j+2], O_frag[4*j+3]);
        }
      }
    }
  }
  // float* O_frag = dynShm + BLK_M*2 + blockDim.x*blockDim.y*4 + warpId*blockDim.x*8;
  if(applySoftmax){
    for(int i = 0; i < 2; i++){
      float rowSum = dynShm[BLK_M + laneId/4 + i*BLK_N ];
      if(rowSum != 0.0f){
        // O_frag[blockDim.x*i*2 + laneId] = O_frag[blockDim.x*i*2 + laneId] * (1.0f/rowSum);
        // O_frag[blockDim.x*(i*2+1) + laneId] = O_frag[blockDim.x*(i*2+1) + laneId] * (1.0f/rowSum);
        // O_frag[blockDim.x*(i*2+4) + laneId] = O_frag[blockDim.x*(i*2+4) + laneId] * (1.0f/rowSum);
        // O_frag[blockDim.x*(i*2+5) + laneId] = O_frag[blockDim.x*(i*2+5) + laneId] * (1.0f/rowSum);
        O_frag[i*2] = O_frag[i*2] * (1.0f/rowSum);
        O_frag[i*2+1] = O_frag[i*2+1] * (1.0f/rowSum);
        O_frag[i*2+4] = O_frag[i*2+4] * (1.0f/rowSum);
        O_frag[i*2+5] = O_frag[i*2+5] * (1.0f/rowSum);
        
      }
    }
  }
  for(int j=0; j < 2; j++){// 2 8x8 blocks in each 16x8 block
    int rowIdx = bid * BLK_M + (laneId / 4) + j * BLK_M/2;
    for(int i =0; i < 2; i++){// 2 16x8 blocks
      int colIdx = (warpId * 2 + i) * BLK_N + (laneId % 4) * 2;
      // output[rowIdx * embeddingDim + colIdx] = O_frag[blockDim.x*(i*4 + j*2) + laneId];
      // output[rowIdx * embeddingDim + colIdx + 1] = O_frag[blockDim.x*(i*4 + j*2 + 1) + laneId]; 
      output[rowIdx * embeddingDim + colIdx] = O_frag[i*4 + j*2];
      output[rowIdx * embeddingDim + colIdx + 1] = O_frag[i*4 + j*2 + 1]; 
    }
  }
}

// load Q from HBM to register. Permute columns
__device__ void loadQFragPermuteCol(volatile uint32_t *Q_frag, uint64_t *Q, int embeddingDim, int rowIdx, int colIdx) {
    uint64_t val = Q[rowIdx * embeddingDim + colIdx];
    Q_frag[0] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
    Q_frag[2] = static_cast<uint32_t>(val >> 32);
    val = Q[(rowIdx+8) * embeddingDim + colIdx];
    Q_frag[1] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
    Q_frag[3] = static_cast<uint32_t>(val >> 32);
}

// ind is in 128b interval
// Assume Q is stored in row-major order. We divide Q into N 16x32 element block. 
__device__ void loadQHbm2Shm128b(uint64_t* qShm, ulonglong2* Q, int embeddingDim, int ind){
  int qWidth = embeddingDim/8;
  ulonglong2 val = Q[ind];
  ulonglong2 val2 = Q[ind + BLK_M/2 * qWidth];
  qWidth *= 2; //convert to 64b width
  ind *= 2; //convert to 64b index
  int rid = ind / qWidth;
  int cid = ind % qWidth;
  // 8 because 32*sizeof(half)/sizeof(uint64_t) = 8
  int blockCid = cid / 8;
  int cidInBlock = cid % 8;
  int blockOffset = blockCid*BLK_M*8 + rid*8 + cidInBlock;
  qShm[blockOffset] = val.x;
  qShm[blockOffset + 1] = val2.x;
  qShm[blockOffset + 64] = val.y;
  qShm[blockOffset + 65] = val2.y;
}

// Pair with loadQHbm2Shm128b. 
// This function has each warp read 1 16x16 block. 
//Not following the register layout in ptx doc but reordering it to match how K is loaded.
__device__ void loadQFragShm(uint64_t* Q_frag, uint64_t* dynShm, int ind, int laneId) {
  // 4 because BLK_M*sizeof(half)/sizeof(uint64_t) = 4
  int offset = ind*BLK_M*4 + laneId*2;
  Q_frag[0] = dynShm[offset];
  Q_frag[1] = dynShm[offset + 1];
}

__device__ void storeEFragShm(float* E_frag, uint32_t* dynShm) {
  Half2Uint32 h2U32Converter;
  for(int i = 0; i < 2; i++){
    h2U32Converter.h2.x = __float2half(E_frag[i*2]);
    h2U32Converter.h2.y = __float2half(E_frag[i*2+1]);
    dynShm[i*32] = h2U32Converter.u32;
  }
}
__device__ void loadEFragShm(uint32_t* E_frag, uint32_t* dynShm) {
  E_frag[0] = dynShm[0];
  E_frag[1] = dynShm[32];
  E_frag[2] = dynShm[64];
  E_frag[3] = dynShm[96];
}

__device__ void loadOFragShm(volatile float* O_frag, float* dynShm, float* mTilde) {
  O_frag[0] = dynShm[0] * mTilde[0];
  O_frag[1] = dynShm[32] * mTilde[0];
  O_frag[2] = dynShm[64] * mTilde[BLK_M/2];
  O_frag[3] = dynShm[96] * mTilde[BLK_M/2];
}

__device__ void storeOFragShm(volatile float* O_frag, float* dynShm) {
  dynShm[0] = O_frag[0];
  dynShm[32] = O_frag[1];
  dynShm[64] = O_frag[2];
  dynShm[96] = O_frag[3];
}

__device__ void printQFrag(uint64_t* Q_frag_uint64, int laneId) {
  if(threadIdx.y == 0){
    Half2Uint32 h2U32Converter0;
    Half2Uint32 h2U32Converter1;
    Half2Uint32 h2U32Converter2;
    Half2Uint32 h2U32Converter3;
    h2U32Converter0.u32 = static_cast<uint32_t>(Q_frag_uint64[0]& 0xFFFFFFFFull);
    h2U32Converter1.u32 = static_cast<uint32_t>(Q_frag_uint64[0] >> 32);
    h2U32Converter2.u32 = static_cast<uint32_t>(Q_frag_uint64[1]& 0xFFFFFFFFull);
    h2U32Converter3.u32 = static_cast<uint32_t>(Q_frag_uint64[1] >> 32);
    printf("laneId: %d, Q0: %f, Q1: %f, Q2: %f, Q3: %f, Q4: %f, Q5: %f, Q6: %f, Q7: %f\n", laneId, __half2float(h2U32Converter0.h2.x), __half2float(h2U32Converter0.h2.y), __half2float(h2U32Converter1.h2.x), __half2float(h2U32Converter1.h2.y), __half2float(h2U32Converter2.h2.x), __half2float(h2U32Converter2.h2.y), __half2float(h2U32Converter3.h2.x), __half2float(h2U32Converter3.h2.y));
  }
}

__device__ void printKFrag(ulonglong2 val, int laneId) {
  Half2Uint32 h2U32Converter0;
  Half2Uint32 h2U32Converter1;
  Half2Uint32 h2U32Converter2;
  Half2Uint32 h2U32Converter3;
  h2U32Converter0.u32 = static_cast<uint32_t>(val.x & 0xFFFFFFFFull);
  h2U32Converter1.u32 = static_cast<uint32_t>(val.x >> 32);
  h2U32Converter2.u32 = static_cast<uint32_t>(val.y & 0xFFFFFFFFull);
  h2U32Converter3.u32 = static_cast<uint32_t>(val.y >> 32);
  printf("warpId: %d, laneId: %d, K0: %f, K1: %f, K2: %f, K3: %f, K4: %f, K5: %f, K6: %f, K7: %f\n", threadIdx.y, laneId, __half2float(h2U32Converter0.h2.x), __half2float(h2U32Converter0.h2.y), __half2float(h2U32Converter1.h2.x), __half2float(h2U32Converter1.h2.y), __half2float(h2U32Converter2.h2.x), __half2float(h2U32Converter2.h2.y), __half2float(h2U32Converter3.h2.x), __half2float(h2U32Converter3.h2.y));
}

// Each warp computes 1 tcb of S.
// TODO: right now we need at least 2 warps per block because we need 2 tcbs to go to the spmm stage.
__global__ void f3sKernel1tb1rw(
    const int *__restrict__ rowWindowOffset,
    const int *__restrict__ sparseAToXidx, 
    const uint64_t *__restrict__ tcbBitMap,
    int embeddingDim,
    ulonglong2 *__restrict__ Q, 
    ulonglong2 *__restrict__ K, 
    half *__restrict__ V,
    float2 *output,
    float2 *sddmmResult) {
  volatile int laneId = threadIdx.x;
  int warpId = threadIdx.y;
  // contains a RW of Q
  extern __shared__ __align__(16) uint64_t dynShm1tb1rw[];
  __shared__ float maxOld[BLK_M];
  // r_b in Alg 1
  __shared__ float sumOld[BLK_M];
  __shared__ float mTilde[BLK_M];
  {
    //initialize everything to 0
    int oOffset = (embeddingDim+blockDim.y*BLK_N)*BLK_M/4 + 2*blockDim.y*BLK_M/2;
    for(int i = tid; i < oOffset + embeddingDim*BLK_M/2; i += blockDim.x*blockDim.y){
      dynShm1tb1rw[i] = 0;
    }
    for(int i = tid; i < BLK_M; i += blockDim.x*blockDim.y){
      maxOld[i] = 0.0f;
      sumOld[i] = 0.0f;
      mTilde[i] = 0.0f;
    }
  }
  //BLK_M/2 because each thread loads 2 128b elements
  for(int i = tid; i < (BLK_M/2)*embeddingDim/8; i += blockDim.x*blockDim.y){
    loadQHbm2Shm128b(dynShm1tb1rw, Q+bid*BLK_M*embeddingDim/8, embeddingDim, i);
  }
  __syncthreads();

  int niter = ((rowWindowOffset[bid+1] - rowWindowOffset[bid])*2 + blockDim.y - 1)/blockDim.y;
  #pragma unroll 1
  for(int iter = 0; iter < niter; iter++){
    int iterTcbStart = rowWindowOffset[bid] + iter*blockDim.y/2;
    // number of 16x16 blocks in S/E being computed in this iteration.
    int nBlock = min(blockDim.y/2, rowWindowOffset[bid+1]-iterTcbStart);
    float S_frag[4] = {0.0f};
    int warpTcbId = warpId/2 + iterTcbStart;
    if(warpId < nBlock*2){
      {//sddmm
        int kOffset = sparseAToXidx[warpTcbId*BLK_M + (warpId%2)*BLK_N + laneId/4] * embeddingDim/8 + laneId % 4;
        for(int i = 0; i < embeddingDim/BLK_K; i+=2) {
          //load K with permuted columns
          ulonglong2 val = K[kOffset + i*BLK_K/8];
          uint64_t Q_frag[2];
          loadQFragShm(Q_frag, dynShm1tb1rw, i, laneId);
          HMMA16816(S_frag[0], S_frag[1], S_frag[2], S_frag[3], 
                    static_cast<uint32_t>(Q_frag[0] & 0xFFFFFFFFull), 
                    static_cast<uint32_t>(Q_frag[1] & 0xFFFFFFFFull), 
                    static_cast<uint32_t>(Q_frag[0] >> 32), 
                    static_cast<uint32_t>(Q_frag[1] >> 32), 
                    static_cast<uint32_t>(val.x & 0xFFFFFFFFull), 
                    static_cast<uint32_t>(val.x >> 32), 
                    S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
          loadQFragShm(Q_frag, dynShm1tb1rw, i+1, laneId);
          HMMA16816(S_frag[0], S_frag[1], S_frag[2], S_frag[3], 
                    static_cast<uint32_t>(Q_frag[0] & 0xFFFFFFFFull), 
                    static_cast<uint32_t>(Q_frag[1] & 0xFFFFFFFFull), 
                    static_cast<uint32_t>(Q_frag[0] >> 32), 
                    static_cast<uint32_t>(Q_frag[1] >> 32), 
                    static_cast<uint32_t>(val.y & 0xFFFFFFFFull), 
                    static_cast<uint32_t>(val.y >> 32), 
                    S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
        }
        int bitIdx = 63 - laneId*2;
        for(int i = 0; i < 4; i++){
          uint64_t bitMask = 1ULL << (bitIdx - i%2);
          S_frag[i] = (tcbBitMap[warpTcbId*4+(warpId%2)*2+i/2] & bitMask) == 0 ? 0.0f : S_frag[i];
        }
      }
      // {//save sddmm result
      //   int offset = warpTcbId*BLK_M*BLK_M + (warpId%2)*BLK_M*BLK_N + laneId*2;
      //   for(int j = 0; j < 2; j++){ // 2 8x8 blocks in each 16x8 block
      //     int sumOffset = j*BLK_N*BLK_N;
      //     float2 val;
      //     val.x = S_frag[j*2];
      //     val.y = S_frag[j*2+1];
      //     sddmmResult[(offset + sumOffset)/2] = val;
      //   }
      // }
      {//online softmax
        float* maxPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2 + laneId/4 + blockDim.y*BLK_M;
        //save max of each row within the warp to shared memory for cross-warp communication
        for(int i=0; i<2; i++){
          float localMax = fmaxf(S_frag[i*2], S_frag[i*2+1]);
          reduceMax(localMax, laneId);
          if(laneId % 4 == 0){
            maxPtr[warpId*BLK_M + i*BLK_M/2] = localMax;
          }
        }
      }
    }
    __syncthreads();
    if(warpId < nBlock*2){
      float* sumPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2;
      for(int i=0; i<2; i++){
        int offset = i*BLK_M/2 + laneId/4;
        float* maxPtr = sumPtr + blockDim.y*BLK_M + offset;
        float newGlobalMax = maxOld[offset];
        //we have blockDim.y columns reserved for local sum of each warp
        //but only nBlock*2 are used
        for(int j=0; j<nBlock*2; j++){
          newGlobalMax = fmaxf(maxPtr[j*BLK_M], newGlobalMax);
        }
        if(warpId == 0 && laneId % 4 == 0){
          mTilde[offset] = __expf(maxOld[offset] - newGlobalMax);
          maxOld[offset] = newGlobalMax;
        }
        //compute E, ignore 0s
        S_frag[i*2] = S_frag[i*2]==0.0f ? 0.0f : __expf(S_frag[i*2] - newGlobalMax);
        S_frag[i*2+1] = S_frag[i*2+1]==0.0f ? 0.0f : __expf(S_frag[i*2+1] - newGlobalMax);
        //compute row sum and save to shared memory
        float localSum = S_frag[i*2] + S_frag[i*2+1];
        reduceSum(localSum);
        if(laneId % 4 == 0){
          sumPtr[warpId*BLK_M + offset] = localSum;
        }
      }
      int eOffset = (embeddingDim + warpId*BLK_N)*BLK_M/2+laneId;
      storeEFragShm(S_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
    }
    __syncthreads();
    //Could try moving this block to the end of the loop
    if(tid < BLK_M){
      float* sumPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2 + laneId;
      //update r_b
      float rowSum = 0.0f;
      //we have blockDim.y columns reserved for local sum of each warp
      //but only nBlock*2 are used
      for(int j=0; j<nBlock*2; j++){
        rowSum += sumPtr[j*BLK_M];
      }
      sumOld[laneId] = fmaf(mTilde[laneId], sumOld[laneId], rowSum);
    }
    __syncthreads();
    {//SpMM
      int oOffset_base = (embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + laneId;
      for(int i=warpId; i<embeddingDim/BLK_N; i+=blockDim.y){
        int oOffset = oOffset_base + i*BLK_M*BLK_N;
        volatile float O_frag[4];
        loadOFragShm(O_frag, reinterpret_cast<float*>(dynShm1tb1rw)+oOffset, mTilde+laneId/4);
        for(int j=0; j<nBlock; j++){
          uint32_t E_frag[4];
          uint32_t B_frag[2];
          //load E
          int eOffset = (embeddingDim+j*BLK_M)*BLK_M/2 + laneId;
          loadEFragShm(E_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
          //load V
          int sparseAToXidxOffset = (iterTcbStart+j)*BLK_M + (laneId%4)*2;
          for(int k = 0; k < 2; k++){
            sparseAToXidxOffset += k*BLK_N;
            Half2Uint32 h2U32Converter;
            int offset = sparseAToXidx[sparseAToXidxOffset]*embeddingDim+ i*BLK_N + laneId/4;
            h2U32Converter.h2.x = V[offset];
            offset = sparseAToXidx[sparseAToXidxOffset+1]*embeddingDim+ i*BLK_N + laneId/4;
            h2U32Converter.h2.y = V[offset];
            B_frag[k] = h2U32Converter.u32;
          }
          HMMA16816(O_frag[0], O_frag[1], O_frag[2], O_frag[3], 
                    E_frag[0], E_frag[1], E_frag[2], E_frag[3], 
                    B_frag[0], B_frag[1], 
                    O_frag[0], O_frag[1], O_frag[2], O_frag[3]);
        }
        storeOFragShm(O_frag, reinterpret_cast<float*>(dynShm1tb1rw)+oOffset);
      }
    }
  }
  __syncthreads();
  float invR0 = sumOld[laneId/4] == 0.0f ? 0.0f : 1.0f/sumOld[laneId/4];
  float invR1 = sumOld[BLK_M/2 + laneId/4] == 0.0f ? 0.0f : 1.0f/sumOld[BLK_M/2 + laneId/4];
  //points to (laneId)th element of O
  int oOffset = (embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + laneId;
  //offset in terms of number of elements,
  //have to be divided by 2 to get the index of the float2
  int outputOffset = (bid*BLK_M + laneId/4)*embeddingDim + (laneId%4)*2;
  for(int i = warpId; i < embeddingDim/BLK_N; i += blockDim.y){
    int offset = oOffset + i*BLK_M*BLK_N;
    float* oPtr = reinterpret_cast<float*>(dynShm1tb1rw) + offset;
    float2 val;
    val.x = oPtr[0] * invR0;
    val.y = oPtr[32] * invR0;
    output[(outputOffset + i*BLK_N)/2] = val;
    val.x = oPtr[64] * invR1;
    val.y = oPtr[96] * invR1;
    output[(outputOffset + i*BLK_N + BLK_M/2*embeddingDim)/2] = val;
  }
}

__global__ void f3sKernel1tb1rwScheduled(
    const int *__restrict__ rowWindowOffset,
    const int *__restrict__ sortedRowWindows,
    const int *__restrict__ sparseAToXidx, 
    const uint64_t *__restrict__ tcbBitMap,
    int embeddingDim,
    int nRw,
    ulonglong2 *__restrict__ Q, 
    ulonglong2 *__restrict__ K, 
    half *__restrict__ V,
    float2 *output,
    float2 *sddmmResult) {
  Scheduler scheduler;
  volatile int laneId = threadIdx.x;
  int warpId = threadIdx.y;
  // contains a RW of Q
  extern __shared__ __align__(16) uint64_t dynShm1tb1rw[];
  __shared__ float maxOld[BLK_M];
  // r_b in Alg 1
  __shared__ float sumOld[BLK_M];
  __shared__ float mTilde[BLK_M];

  while(scheduler.next_iter(sortedRowWindows, nRw)){
    int niter = ((rowWindowOffset[scheduler.targetRw+1] - rowWindowOffset[scheduler.targetRw])*2
                 + blockDim.y - 1)/blockDim.y;
    {
      //initialize everything to 0
      int oOffset = (embeddingDim+blockDim.y*BLK_N)*BLK_M/4 + 2*blockDim.y*BLK_M/2;
      for(int i = tid; i < oOffset + embeddingDim*BLK_M/2; i += blockDim.x*blockDim.y){
        dynShm1tb1rw[i] = 0;
      }
      for(int i = tid; i < BLK_M; i += blockDim.x*blockDim.y){
        maxOld[i] = 0.0f;
        sumOld[i] = 0.0f;
        mTilde[i] = 0.0f;
      }
    }
    //BLK_M/2 because each thread loads 2 128b elements
    for(int i = tid; i < (BLK_M/2)*embeddingDim/8; i += blockDim.x*blockDim.y){
      loadQHbm2Shm128b(dynShm1tb1rw, Q+scheduler.targetRw*BLK_M*embeddingDim/8, embeddingDim, i);
    }
    __syncthreads();
    #pragma unroll 1
    for(int iter = 0; iter < niter; iter++){
      int iterTcbStart = rowWindowOffset[scheduler.targetRw] + iter*blockDim.y/2;
      // number of 16x16 blocks in S/E being computed in this iteration.
      int nBlock = min(blockDim.y/2, rowWindowOffset[scheduler.targetRw+1]-iterTcbStart);
      float S_frag[4] = {0.0f};
      int warpTcbId = warpId/2 + iterTcbStart;
      if(warpId < nBlock*2){
        {//sddmm
          int kOffset = sparseAToXidx[warpTcbId*BLK_M + (warpId%2)*BLK_N + laneId/4] 
                        * embeddingDim/8 + laneId % 4;
          for(int i = 0; i < embeddingDim/BLK_K; i+=2) {
            //load K with permuted columns
            ulonglong2 val = K[kOffset + i*BLK_K/8];
            uint64_t Q_frag[2];
            loadQFragShm(Q_frag, dynShm1tb1rw, i, laneId);
            HMMA16816(S_frag[0], S_frag[1], S_frag[2], S_frag[3], 
                      static_cast<uint32_t>(Q_frag[0] & 0xFFFFFFFFull), 
                      static_cast<uint32_t>(Q_frag[1] & 0xFFFFFFFFull), 
                      static_cast<uint32_t>(Q_frag[0] >> 32), 
                      static_cast<uint32_t>(Q_frag[1] >> 32), 
                      static_cast<uint32_t>(val.x & 0xFFFFFFFFull), 
                      static_cast<uint32_t>(val.x >> 32), 
                      S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
            loadQFragShm(Q_frag, dynShm1tb1rw, i+1, laneId);
            HMMA16816(S_frag[0], S_frag[1], S_frag[2], S_frag[3], 
                      static_cast<uint32_t>(Q_frag[0] & 0xFFFFFFFFull), 
                      static_cast<uint32_t>(Q_frag[1] & 0xFFFFFFFFull), 
                      static_cast<uint32_t>(Q_frag[0] >> 32), 
                      static_cast<uint32_t>(Q_frag[1] >> 32), 
                      static_cast<uint32_t>(val.y & 0xFFFFFFFFull), 
                      static_cast<uint32_t>(val.y >> 32), 
                      S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
          }
          int bitIdx = 63 - laneId*2;
          for(int i = 0; i < 4; i++){
            uint64_t bitMask = 1ULL << (bitIdx - i%2);
            S_frag[i] = (tcbBitMap[warpTcbId*4+(warpId%2)*2+i/2] & bitMask) == 0 ? 0.0f : S_frag[i];
          }
        }
        // {//save sddmm result
        //   int offset = warpTcbId*BLK_M*BLK_M + (warpId%2)*BLK_M*BLK_N + laneId*2;
        //   for(int j = 0; j < 2; j++){ // 2 8x8 blocks in each 16x8 block
        //     int sumOffset = j*BLK_N*BLK_N;
        //     float2 val;
        //     val.x = S_frag[j*2];
        //     val.y = S_frag[j*2+1];
        //     sddmmResult[(offset + sumOffset)/2] = val;
        //   }
        // }
        {//online softmax
          float* maxPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2 + laneId/4 + blockDim.y*BLK_M;
          //save max of each row within the warp to shared memory for cross-warp communication
          for(int i=0; i<2; i++){
            float localMax = fmaxf(S_frag[i*2], S_frag[i*2+1]);
            reduceMax(localMax, laneId);
            if(laneId % 4 == 0){
              maxPtr[warpId*BLK_M + i*BLK_M/2] = localMax;
            }
          }
        }
      }
      __syncthreads();
      if(warpId < nBlock*2){
        float* sumPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2;
        for(int i=0; i<2; i++){
          int offset = i*BLK_M/2 + laneId/4;
          float* maxPtr = sumPtr + blockDim.y*BLK_M + offset;
          float newGlobalMax = maxOld[offset];
          //we have blockDim.y columns reserved for local sum of each warp
          //but only nBlock*2 are used
          for(int j=0; j<nBlock*2; j++){
            newGlobalMax = fmaxf(maxPtr[j*BLK_M], newGlobalMax);
          }
          if(warpId == 0 && laneId % 4 == 0){
            mTilde[offset] = __expf(maxOld[offset] - newGlobalMax);
            maxOld[offset] = newGlobalMax;
          }
          //compute E, ignore 0s
          S_frag[i*2] = S_frag[i*2]==0.0f ? 0.0f : __expf(S_frag[i*2] - newGlobalMax);
          S_frag[i*2+1] = S_frag[i*2+1]==0.0f ? 0.0f : __expf(S_frag[i*2+1] - newGlobalMax);
          //compute row sum and save to shared memory
          float localSum = S_frag[i*2] + S_frag[i*2+1];
          reduceSum(localSum);
          if(laneId % 4 == 0){
            sumPtr[warpId*BLK_M + offset] = localSum;
          }
        }
        int eOffset = (embeddingDim + warpId*BLK_N)*BLK_M/2+laneId;
        storeEFragShm(S_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
      }
      __syncthreads();
      //Could try moving this block to the end of the loop
      if(tid < BLK_M){
        float* sumPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2 + laneId;
        //update r_b
        float rowSum = 0.0f;
        //we have blockDim.y columns reserved for local sum of each warp
        //but only nBlock*2 are used
        for(int j=0; j<nBlock*2; j++){
          rowSum += sumPtr[j*BLK_M];
        }
        sumOld[laneId] = fmaf(mTilde[laneId], sumOld[laneId], rowSum);
      }
      __syncthreads();
      {//SpMM
        int oOffset_base = (embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + laneId;
        for(int i=warpId; i<embeddingDim/BLK_N; i+=blockDim.y){
          int oOffset = oOffset_base + i*BLK_M*BLK_N;
          volatile float O_frag[4];
          loadOFragShm(O_frag, reinterpret_cast<float*>(dynShm1tb1rw)+oOffset, mTilde+laneId/4);
          for(int j=0; j<nBlock; j++){
            uint32_t E_frag[4];
            uint32_t B_frag[2];
            //load E
            int eOffset = (embeddingDim+j*BLK_M)*BLK_M/2 + laneId;
            loadEFragShm(E_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
            //load V
            int sparseAToXidxOffset = (iterTcbStart+j)*BLK_M + (laneId%4)*2;
            for(int k = 0; k < 2; k++){
              sparseAToXidxOffset += k*BLK_N;
              Half2Uint32 h2U32Converter;
              int offset = sparseAToXidx[sparseAToXidxOffset]*embeddingDim+ i*BLK_N + laneId/4;
              h2U32Converter.h2.x = V[offset];
              offset = sparseAToXidx[sparseAToXidxOffset+1]*embeddingDim+ i*BLK_N + laneId/4;
              h2U32Converter.h2.y = V[offset];
              B_frag[k] = h2U32Converter.u32;
            }
            HMMA16816(O_frag[0], O_frag[1], O_frag[2], O_frag[3], 
                      E_frag[0], E_frag[1], E_frag[2], E_frag[3], 
                      B_frag[0], B_frag[1], 
                      O_frag[0], O_frag[1], O_frag[2], O_frag[3]);
          }
          storeOFragShm(O_frag, reinterpret_cast<float*>(dynShm1tb1rw)+oOffset);
        }
      }
    }
    __syncthreads();
    float invR0 = sumOld[laneId/4] == 0.0f ? 0.0f : __frcp_rn(sumOld[laneId/4]);
    float invR1 = sumOld[BLK_M/2 + laneId/4] == 0.0f ? 0.0f : __frcp_rn(sumOld[BLK_M/2 + laneId/4]);
    //points to (laneId)th element of O
    int oOffset = (embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + laneId;
    //offset in terms of number of elements,
    //have to be divided by 2 to get the index of the float2
    int outputOffset = (scheduler.targetRw*BLK_M + laneId/4)*embeddingDim + (laneId%4)*2;
    for(int i = warpId; i < embeddingDim/BLK_N; i += blockDim.y){
      int offset = oOffset + i*BLK_M*BLK_N;
      float* oPtr = reinterpret_cast<float*>(dynShm1tb1rw) + offset;
      float2 val;
      val.x = oPtr[0] * invR0;
      val.y = oPtr[32] * invR0;
      output[(outputOffset + i*BLK_N)/2] = val;
      val.x = oPtr[64] * invR1;
      val.y = oPtr[96] * invR1;
      output[(outputOffset + i*BLK_N + BLK_M/2*embeddingDim)/2] = val;
    }
  }
}

// Each warp computes 1 tcb of S.
// TODO: right now we need at least 2 warps per block because we need 2 tcbs to go to the spmm stage.
__global__ void f2sKernel1tb1rw(
    const int *__restrict__ rowWindowOffset,
    const int *__restrict__ sparseAToXidx, 
    const uint64_t *__restrict__ tcbBitMap,
    int embeddingDim,
    ulonglong2 *__restrict__ Q, 
    ulonglong2 *__restrict__ K, 
    half *__restrict__ V,
    float2 *output,
    float2 *sddmmResult) {
  volatile int laneId = threadIdx.x;
  int warpId = threadIdx.y;
  // contains a RW of Q
  extern __shared__ __align__(16) uint64_t dynShm1tb1rw[];
  __shared__ float mTilde[BLK_M];
  {
    //initialize everything to 0
    int oOffset = (embeddingDim+blockDim.y*BLK_N)*BLK_M/4 + 2*blockDim.y*BLK_M/2;
    for(int i = tid; i < oOffset + embeddingDim*BLK_M/2; i += blockDim.x*blockDim.y){
      dynShm1tb1rw[i] = 0;
    }
    for(int i = tid; i < BLK_M; i += blockDim.x*blockDim.y){
      mTilde[i] = 1.0f;
    }
  }
  //BLK_M/2 because each thread loads 2 128b elements
  for(int i = tid; i < (BLK_M/2)*embeddingDim/8; i += blockDim.x*blockDim.y){
    loadQHbm2Shm128b(dynShm1tb1rw, Q+bid*BLK_M*embeddingDim/8, embeddingDim, i);
  }
  __syncthreads();

  int niter = ((rowWindowOffset[bid+1] - rowWindowOffset[bid]) + (blockDim.y/2) - 1)/(blockDim.y/2);
  #pragma unroll 1
  for(int iter = 0; iter < niter; iter++){
    int iterTcbStart = rowWindowOffset[bid] + iter*(blockDim.y/2);
    // number of 16x16 blocks in S/E being computed in this iteration.
    // This is a check in case the last iteration is not full
    int nBlock = min(blockDim.y/2, rowWindowOffset[bid+1]-iterTcbStart);
    float S_frag[4] = {0.0f};
    int warpTcbId = warpId/2 + iterTcbStart;
    __syncthreads();//DEBUG
    if(warpTcbId < rowWindowOffset[bid+1]){
      {//sddmm
        int kOffset = sparseAToXidx[warpTcbId*BLK_M + (warpId%2)*BLK_N + laneId/4] * embeddingDim/8 + laneId % 4;
        for(int i = 0; i < embeddingDim/BLK_K; i+=2) {
          //load K with permuted columns
          ulonglong2 val = K[kOffset + i*BLK_K/8];
          uint64_t Q_frag[2];
          loadQFragShm(Q_frag, dynShm1tb1rw, i, laneId);
          HMMA16816(S_frag[0], S_frag[1], S_frag[2], S_frag[3], 
                    static_cast<uint32_t>(Q_frag[0] & 0xFFFFFFFFull), 
                    static_cast<uint32_t>(Q_frag[1] & 0xFFFFFFFFull), 
                    static_cast<uint32_t>(Q_frag[0] >> 32), 
                    static_cast<uint32_t>(Q_frag[1] >> 32), 
                    static_cast<uint32_t>(val.x & 0xFFFFFFFFull), 
                    static_cast<uint32_t>(val.x >> 32), 
                    S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
          loadQFragShm(Q_frag, dynShm1tb1rw, i+1, laneId);
          HMMA16816(S_frag[0], S_frag[1], S_frag[2], S_frag[3], 
                    static_cast<uint32_t>(Q_frag[0] & 0xFFFFFFFFull), 
                    static_cast<uint32_t>(Q_frag[1] & 0xFFFFFFFFull), 
                    static_cast<uint32_t>(Q_frag[0] >> 32), 
                    static_cast<uint32_t>(Q_frag[1] >> 32), 
                    static_cast<uint32_t>(val.y & 0xFFFFFFFFull), 
                    static_cast<uint32_t>(val.y >> 32), 
                    S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
        }
        int bitIdx = 63 - laneId*2;
        for(int i = 0; i < 4; i++){
          uint64_t bitMask = 1ULL << (bitIdx - i%2);
          S_frag[i] = (tcbBitMap[warpTcbId*4+(warpId%2)*2+i/2] & bitMask) == 0 ? 0.0f : S_frag[i];
        }
      }
      {//save sddmm result
        int offset = warpTcbId*BLK_M*BLK_M + (warpId%2)*BLK_M*BLK_N + laneId*2;
        for(int j = 0; j < 2; j++){ // 2 8x8 blocks in each 16x8 block
          int sumOffset = j*BLK_N*BLK_N;
          float2 val;
          val.x = S_frag[j*2];
          val.y = S_frag[j*2+1];
          sddmmResult[(offset + sumOffset)/2] = val;
        }
      }
    }
    __syncthreads();
    if(warpTcbId < rowWindowOffset[bid+1]){
      int eOffset = (embeddingDim + warpId*BLK_N)*BLK_M/2+laneId;
      storeEFragShm(S_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
    }
    __syncthreads();
    {//SpMM
      int oOffset_base = (embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + laneId;
      for(int i=warpId; i<embeddingDim/BLK_N; i+=blockDim.y){
        int oOffset = oOffset_base + i*BLK_M*BLK_N;
        float O_frag[4];
        loadOFragShm(O_frag, reinterpret_cast<float*>(dynShm1tb1rw)+oOffset, mTilde+laneId/4);
        for(int j=0; j<nBlock; j++){
          uint32_t E_frag[4];
          uint32_t B_frag[2];
          //load E
          int eOffset = (embeddingDim+j*BLK_M)*BLK_M/2 + laneId;
          loadEFragShm(E_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
          //load V
          int sparseAToXidxOffset = (iterTcbStart+j)*BLK_M + (laneId%4)*2;
          for(int k = 0; k < 2; k++){
            sparseAToXidxOffset += k*BLK_N;
            Half2Uint32 h2U32Converter;
            int offset = sparseAToXidx[sparseAToXidxOffset]*embeddingDim+ i*BLK_N + laneId/4;
            h2U32Converter.h2.x = V[offset];
            offset = sparseAToXidx[sparseAToXidxOffset+1]*embeddingDim+ i*BLK_N + laneId/4;
            h2U32Converter.h2.y = V[offset];
            B_frag[k] = h2U32Converter.u32;
          }
          HMMA16816(O_frag[0], O_frag[1], O_frag[2], O_frag[3], 
                    E_frag[0], E_frag[1], E_frag[2], E_frag[3], 
                    B_frag[0], B_frag[1], 
                    O_frag[0], O_frag[1], O_frag[2], O_frag[3]);
        }
        storeOFragShm(O_frag, reinterpret_cast<float*>(dynShm1tb1rw)+oOffset);
      }
    }
    __syncthreads();
  }
  __syncthreads();
  //points to (laneId)th element of O
  int oOffset = (embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + laneId;
  //offset in terms of number of elements,
  //have to be divided by 2 to get the index of the float2
  int outputOffset = (bid*BLK_M + laneId/4)*embeddingDim + (laneId%4)*2;
  for(int i = warpId; i < embeddingDim/BLK_N; i += blockDim.y){
    int offset = oOffset + i*BLK_M*BLK_N;
    float* oPtr = reinterpret_cast<float*>(dynShm1tb1rw) + offset;
    float2 val;
    val.x = oPtr[0];
    val.y = oPtr[32];
    output[(outputOffset + i*BLK_N)/2] = val;
    val.x = oPtr[64];
    val.y = oPtr[96];
    output[(outputOffset + i*BLK_N + BLK_M/2*embeddingDim)/2] = val;
  }
}

// Each tb computes multiple row windows of S
__global__ void sddmmKernel1tbnrw(
    const int *__restrict__ rowWindowOffset,
    const int *__restrict__ tbBoundaries,
    const int *__restrict__ tcbRowId,
    const int *__restrict__ sparseAToXidx, 
    const uint64_t *__restrict__ tcbBitMap,
    int embeddingDim,
    torch::Half *__restrict__ Q, 
    torch::Half *__restrict__ K, 
    float2 *output) {
  volatile int laneId = threadIdx.x;
  int warpId = threadIdx.y;
  volatile int tidInGroup = laneId % 4;
  int tcbStart = rowWindowOffset[tbBoundaries[bid]];
  int tcbEnd = rowWindowOffset[tbBoundaries[bid+1]];
  for(int tcbId = tcbStart + warpId; tcbId < tcbEnd; tcbId += blockDim.y) {
    volatile float S_frag[4] = {0.0f};
    volatile uint32_t Q_frag[4];
    volatile uint32_t K_frag[2];
    int rowIdxQ = tcbRowId[tcbId]*BLK_M + laneId/4;
    int rowIdxK = sparseAToXidx[tcbId*BLK_N + laneId/4] * embeddingDim/4; 
    for(int i = 0; i < embeddingDim/BLK_K; i++) {
      loadQFragPermuteCol(Q_frag, reinterpret_cast<uint64_t*>(Q), embeddingDim/4, rowIdxQ, i*BLK_K/4 + tidInGroup);
      uint64_t val = reinterpret_cast<uint64_t*>(K)[rowIdxK + i*BLK_K/4 + tidInGroup];
      K_frag[0] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
      K_frag[1] = static_cast<uint32_t>(val >> 32);
      HMMA16816(S_frag[0], S_frag[1], S_frag[2], S_frag[3], 
                Q_frag[0], Q_frag[1], Q_frag[2], Q_frag[3], 
                K_frag[0], K_frag[1], 
                S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
    }
    int bitIdx = 63 - laneId*2;
    for(int i = 0; i < 4; i++){
      uint64_t bitMask = 1ULL << (bitIdx - i%2);
      S_frag[i] = (tcbBitMap[tcbId*2+i/2] & bitMask) == 0 ? 0.0f : S_frag[i];
    }

    int offset = tcbId * BLK_M * BLK_N;
    for(int j = 0; j < 2; j++){ // 2 8x8 blocks in each 16x8 block
      int sumOffset = j*BLK_N*BLK_N + laneId*2;
      float2 val;
      val.x = S_frag[j*2];
      val.y = S_frag[j*2+1];
      output[(offset + sumOffset)/2] = val;
    }
  }
}
// #endif