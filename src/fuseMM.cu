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