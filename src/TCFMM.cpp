#include <torch/extension.h>
#include <vector>
#include <string.h>
#include <cstdlib>
#include <map>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/unique.h>

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void fill_edgeToRow_cuda(int *edgeToRow, int *nodePointer, int num_nodes);

void fill_window_cuda(int *edgeToColumn, int *blockPartition, int *nodePointer,
                      int *edgeList, int blockSize_h, int blockSize_w,
                      int num_nodes);

void fill_segment_cuda(int *nodePointer, int *seg_out, int blockSize_h,
                       int blockSize_w, int num_nodes);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
seg_sort_dequ(int *seg, int *edgeLists, int *nodepointer, int *edgetocol,
              int *edgetorow, int *blockPartition, int *blocknum,
              int *row_window_offset, int blockSize_h, int blockSize_w,
              int num_nodes, int num_edges, int rowwindow_num);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, int>
preprocess_gpu(torch::Tensor edgeList_tensor, torch::Tensor nodePointer_tensor,
               int num_nodes, int blockSize_h, int blockSize_w,
               torch::Tensor blockPartition_tensor,
               torch::Tensor edgeToColumn_tensor,
               torch::Tensor edgeToRow_tensor) {
  // input tensors.
  auto options_gpu =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  auto edgeList = edgeList_tensor.data_ptr<int>();
  auto blockPartition = blockPartition_tensor.data_ptr<int>();
  auto row_window_offset_tensor =
      torch::zeros({blockPartition_tensor.size(0) + 1}, options_gpu);
  auto row_window_offset = row_window_offset_tensor.data_ptr<int>();
  auto edgeToColumn = edgeToColumn_tensor.data_ptr<int>();
  auto seg_out_tensor = torch::zeros({edgeList_tensor.size(0)}, options_gpu);
  auto blocknum = torch::zeros({1}, options_gpu);
  auto block_num = blocknum.data_ptr<int>();
  auto edgeToRow = edgeToRow_tensor.data_ptr<int>();
  auto nodePointer = nodePointer_tensor.data_ptr<int>();
  auto seg_out = seg_out_tensor.data_ptr<int>();
  auto start = std::chrono::high_resolution_clock::now();
  fill_edgeToRow_cuda(edgeToRow, nodePointer, num_nodes);
  int block_counter = 0;
  fill_segment_cuda(nodePointer, seg_out, blockSize_h, blockSize_w, num_nodes);
  auto tuple_tensor_blockcnt = seg_sort_dequ(
      seg_out, edgeList, nodePointer, edgeToColumn, edgeToRow, blockPartition,
      block_num, row_window_offset, blockSize_h, blockSize_w, num_nodes,
      edgeList_tensor.size(0), blockPartition_tensor.size(0));
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "\t GPU Preprocess time: " << elapsed_seconds.count()
            << " seconds\n";
  auto tcblock_offset_tensor = std::get<0>(tuple_tensor_blockcnt);
  auto tcblock_rowid_tensor = std::get<1>(tuple_tensor_blockcnt);
  auto tcblocktile_id_tensor = std::get<2>(tuple_tensor_blockcnt);
  auto sparse_AToX_index_tensor = std::get<3>(tuple_tensor_blockcnt);
  block_counter = std::get<4>(tuple_tensor_blockcnt);
  printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", block_counter,
         block_counter * 8 * 16);
  return std::make_tuple(row_window_offset_tensor, tcblock_rowid_tensor,
                         tcblocktile_id_tensor, tcblock_offset_tensor,
                         sparse_AToX_index_tensor, block_counter);
}


std::vector<torch::Tensor> 
fusedMM_forward_cuda(
	torch::Tensor TCblock_rowid, torch::Tensor TCblocktile_id,
  torch::Tensor TCblock_offset, torch::Tensor sparse_AToX_idx,
  int num_nodes, int num_edges,
  int embedding_dim,  // embedding dimension.
  torch::Tensor input, // input feature matrix.
  bool save_edge_attention,
  bool use_f32_edge_attention,
  bool use_m8n32k16
);

std::vector<torch::Tensor> 
f3S_forward_cuda(
  torch::Tensor TCblock_rowid,
  torch::Tensor sparse_AToX_idx, 
  int num_nodes, 
  int embedding_dim,
  torch::Tensor Q, torch::Tensor K, torch::Tensor V, 
  bool save_sddmm_result
);

std::vector<torch::Tensor>
fusedMM_forward(torch::Tensor input, torch::Tensor TCblock_rowid,
                torch::Tensor TCblocktile_id,
                torch::Tensor TCblock_offset,
                torch::Tensor sparse_AToX_idx, int num_nodes, 
                bool save_edge_attention,
                bool use_f32_edge_attention,
                bool use_m8n32k16) {
  CHECK_INPUT(input);
  CHECK_INPUT(TCblock_rowid);
  CHECK_INPUT(TCblocktile_id);
  CHECK_INPUT(TCblock_offset);
  CHECK_INPUT(sparse_AToX_idx);

  int num_edges = TCblocktile_id.size(0);
  int embedding_dim = input.size(1);

  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  auto result = fusedMM_forward_cuda(TCblock_rowid, TCblocktile_id, 
                                     TCblock_offset, sparse_AToX_idx, 
                                     num_nodes, num_edges, 
                                     embedding_dim, 
                                     input, 
                                     save_edge_attention,
                                     use_f32_edge_attention,
                                     use_m8n32k16);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("fusedMM execution time (cuda events): %f ms\n", elapsedTime);
  return result;
}

std::vector<torch::Tensor>
f3S_forward(torch::Tensor TCblock_rowid,
            torch::Tensor sparse_AToX_idx, 
            int num_nodes, 
            torch::Tensor Q, torch::Tensor K, torch::Tensor V, 
            bool save_sddmm_result) {
  CHECK_INPUT(Q);
  CHECK_INPUT(K);
  CHECK_INPUT(V);
  CHECK_INPUT(TCblock_rowid);
  CHECK_INPUT(sparse_AToX_idx);
  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int embedding_dim = Q.size(1);
  cudaEventRecord(start, 0);
  auto result = f3S_forward_cuda(TCblock_rowid, 
                                 sparse_AToX_idx, 
                                 num_nodes, 
                                 embedding_dim, 
                                 Q, K, V, 
                                 save_sddmm_result);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("f3S forward execution time (cuda events): %f ms\n", elapsedTime);
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess_gpu", &preprocess_gpu, "Preprocess Step on (CUDA)");
  m.def("fusedMM_forward", &fusedMM_forward, "FusedMM forward (CUDA)");
  m.def("f3S_forward", &f3S_forward, "fused Spmm-Softmax-Spmm (CUDA)");
}