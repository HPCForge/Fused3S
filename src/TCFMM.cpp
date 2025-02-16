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

std::vector<int> assign_row_to_tb(int *rowwindow_offset, int nRowWindows);

void fill_edgeToRow_cuda(int *edgeToRow, int *nodePointer, int num_nodes);

void fill_window_cuda(int *edgeToColumn, int *blockPartition, int *nodePointer,
                      int *edgeList, int blockSize_h, int blockSize_w,
                      int num_nodes);

void fill_segment_cuda(int *nodePointer, int *seg_out, int blockSize_h,
                       int blockSize_w, int num_nodes);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
seg_sort_dequ(int *seg, int *edgeLists, int *nodepointer, int *edgetocol,
              int *edgetorow, int *blockPartition, int *blocknum,
              int *row_window_offset, int blockSize_h, int blockSize_w,
              int num_nodes, int num_edges, int rowwindow_num);
              
torch::Tensor 
sort_row_windows_by_tcb_count(const int* rowwindow_offset, int num_row_windows);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
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
  auto sorted_row_window_tensor = sort_row_windows_by_tcb_count(
      row_window_offset_tensor.data_ptr<int>(), blockPartition_tensor.size(0));
  auto row_window_offset_tensor_cpu = row_window_offset_tensor.to(torch::kCPU);
  auto tb_boundaries = assign_row_to_tb(
    row_window_offset_tensor_cpu.data_ptr<int>(), 
    blockPartition_tensor.size(0));
  auto tb_boundaries_tensor = torch::tensor(tb_boundaries, options_gpu);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "\t GPU Preprocess time: " 
            << elapsed_seconds.count()
            << " seconds\n";
  auto tcblock_offset_tensor = std::get<0>(tuple_tensor_blockcnt);
  auto tcblock_rowid_tensor = std::get<1>(tuple_tensor_blockcnt);
  auto tcblocktile_id_tensor = std::get<2>(tuple_tensor_blockcnt);
  auto sparse_AToX_index_tensor = std::get<3>(tuple_tensor_blockcnt);
  auto tcblock_bit_map_tensor = std::get<4>(tuple_tensor_blockcnt);
  block_counter = std::get<5>(tuple_tensor_blockcnt);
  printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", block_counter,
         block_counter * 8 * 16);
  return std::make_tuple(row_window_offset_tensor, sorted_row_window_tensor, 
                         tcblock_rowid_tensor, tcblocktile_id_tensor, 
                         tcblock_offset_tensor, sparse_AToX_index_tensor, 
                         tb_boundaries_tensor, tcblock_bit_map_tensor, 
                         block_counter);
}

std::vector<torch::Tensor>
f3sCuda1tb1tcb(
    torch::Tensor rowWindowOffset,
    torch::Tensor sparseAToXidx, 
    torch::Tensor tcbBitMap,
    int numNodes, 
    int embeddingDim,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, 
    bool applySoftmax,
    bool saveSddmmResult);

std::vector<torch::Tensor> 
f3sCuda1tb1rwClocked(
    torch::Tensor rowWindowOffset,
    torch::Tensor sparseAToXidx,
    torch::Tensor tcbBitMap,
    int nNodes,
    int embeddingDim,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int nWarpPerBlock);

std::vector<torch::Tensor> 
f3sCuda1tb1rw(
    torch::Tensor rowWindowOffset,
    torch::Tensor sparseAToXidx,
    torch::Tensor tcbBitMap,
    int nNodes,
    int embeddingDim,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int nWarpPerBlock,
    bool applySoftmax);

std::vector<torch::Tensor> 
f3sCuda1tb1rwScheduledClocked(
    torch::Tensor rowWindowOffset,
    torch::Tensor sortedRowWindows,
    torch::Tensor sparseAToXidx,
    torch::Tensor tcbBitMap,
    int nNodes,
    int embeddingDim,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int nWarpPerBlock);

std::vector<torch::Tensor> 
f3sCuda1tb1rwScheduled(
    torch::Tensor rowWindowOffset,
    torch::Tensor sortedRowWindows,
    torch::Tensor sparseAToXidx,
    torch::Tensor tcbBitMap,
    int nNodes,
    int embeddingDim,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int nWarpPerBlock,
    bool permuteV);

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
    int nWarpPerBlock);

std::vector<torch::Tensor>
f3s1tb1rw(
  torch::Tensor rowWindowOffset,
  torch::Tensor sparseAToXidx, 
  torch::Tensor tcbBitMap,
  int nNodes, 
  torch::Tensor Q, torch::Tensor K, torch::Tensor V,
  int nWarpPerBlock,
  bool applySoftmax,
  bool checkSMActiveTime
){
  CHECK_INPUT(rowWindowOffset);
  CHECK_INPUT(sparseAToXidx);
  CHECK_INPUT(tcbBitMap);
  CHECK_INPUT(Q);
  CHECK_INPUT(K);
  CHECK_INPUT(V);
  int embeddingDim = Q.size(1);
  std::vector<torch::Tensor> result;
  if(checkSMActiveTime){
    result = f3sCuda1tb1rwClocked(rowWindowOffset, 
                                  sparseAToXidx, 
                                  tcbBitMap, 
                                  nNodes, 
                                  embeddingDim,
                                  Q, K, V,
                                  nWarpPerBlock);
  }else{
    result = f3sCuda1tb1rw(rowWindowOffset, 
                          sparseAToXidx, 
                          tcbBitMap, 
                          nNodes, 
                          embeddingDim,
                          Q, K, V,
                          nWarpPerBlock,
                          applySoftmax);
  }
  return result;
}

std::vector<torch::Tensor>
f3s1tb1rwScheduled(
    torch::Tensor rowWindowOffset,
    torch::Tensor sortedRowWindows,
    torch::Tensor sparseAToXidx,
    torch::Tensor tcbBitMap,
    int nNodes,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int nWarpPerBlock,
    bool checkSMActiveTime){
  int embeddingDim = Q.size(1);
  std::vector<torch::Tensor> result;
  if(checkSMActiveTime){
    result = f3sCuda1tb1rwScheduledClocked(rowWindowOffset, 
                                  sortedRowWindows, 
                                  sparseAToXidx, 
                                  tcbBitMap, 
                                  nNodes, 
                                  embeddingDim, 
                                  Q, K, V, 
                                  nWarpPerBlock);
  }else{
    result = f3sCuda1tb1rwScheduled(rowWindowOffset, 
                                  sortedRowWindows, 
                                  sparseAToXidx, 
                                  tcbBitMap, 
                                  nNodes, 
                                  embeddingDim, 
                                  Q, K, V, 
                                  nWarpPerBlock,
                                  false);
  }
  return result;
}
// same as f3s1tb1rwScheduled except permute V = true
std::vector<torch::Tensor>
f3s1tb1rwScheduledPermuteV(
    torch::Tensor rowWindowOffset,
    torch::Tensor sortedRowWindows,
    torch::Tensor sparseAToXidx,
    torch::Tensor tcbBitMap,
    int nNodes,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int nWarpPerBlock){
  int embeddingDim = Q.size(1);
  std::vector<torch::Tensor> result;
  result = f3sCuda1tb1rwScheduled(rowWindowOffset, 
                                  sortedRowWindows, 
                                  sparseAToXidx, 
                                  tcbBitMap, 
                                  nNodes, 
                                  embeddingDim, 
                                  Q, K, V, 
                                  nWarpPerBlock,
                                  true);
  return result;
}

std::vector<torch::Tensor>
sddmm1tbnrw(
  torch::Tensor rowWindowOffset,
  torch::Tensor tbBoundaries,
  torch::Tensor tcbRowid,
  torch::Tensor sparseAToXidx, 
  torch::Tensor tcbBitMap,
  int nNodes, 
  torch::Tensor Q, torch::Tensor K,
  int nWarpPerBlock
){
  CHECK_INPUT(rowWindowOffset);
  CHECK_INPUT(tbBoundaries);
  CHECK_INPUT(tcbRowid);
  CHECK_INPUT(sparseAToXidx);
  CHECK_INPUT(tcbBitMap);
  CHECK_INPUT(Q);
  CHECK_INPUT(K);
  int embeddingDim = Q.size(1);
  std::vector<torch::Tensor> result;
  result = sddmmCuda1tbnrw(rowWindowOffset, 
                           tbBoundaries, 
                           tcbRowid, 
                           sparseAToXidx, 
                           tcbBitMap, 
                           nNodes, 
                           embeddingDim,
                           Q, K,
                           nWarpPerBlock);
  return result;
}

std::vector<torch::Tensor>
f3s1tb1tcb(torch::Tensor rowWindowOffset,
          torch::Tensor sparseAToXidx, 
          torch::Tensor tcbBitMap,
          int nNodes, 
          torch::Tensor Q, torch::Tensor K, torch::Tensor V, 
          bool applySoftmax,
          bool saveSddmmResult) {
  CHECK_INPUT(Q);
  CHECK_INPUT(K);
  CHECK_INPUT(V);
  CHECK_INPUT(rowWindowOffset);
  CHECK_INPUT(sparseAToXidx);
  CHECK_INPUT(tcbBitMap);
  int embeddingDim = Q.size(1);
  auto result = f3sCuda1tb1tcb(rowWindowOffset, 
                               sparseAToXidx, 
                               tcbBitMap,
                               nNodes, 
                               embeddingDim, 
                               Q, K, V, 
                               applySoftmax,
                               saveSddmmResult);
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess_gpu", &preprocess_gpu, "Preprocess Step on (CUDA)");
  // m.def("fusedMM_forward", &fusedMM_forward, "FusedMM forward (CUDA)");
  m.def("f3s_1tb1tcb", &f3s1tb1tcb, "fused3S 1tb1tcb");
  m.def("f3s_1tb1rw", &f3s1tb1rw, "fused3S 1tb1rw");
  m.def("f3s_1tb1rw_scheduled", &f3s1tb1rwScheduled, "fused3S 1tb1rw scheduled");
  m.def("f3s_1tb1rw_scheduled_permuteV", &f3s1tb1rwScheduledPermuteV, "fused3S 1tb1rw scheduled permuteV");
  m.def("sddmm_1tbnrw", &sddmm1tbnrw, "sddmm 1tbnrw");
}
