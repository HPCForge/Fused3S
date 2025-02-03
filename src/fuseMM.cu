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

// Assume each warp has a 16x8 fp16 matrix in row-major order distributed among threads
// Each thread starts with 2 consecutive fp16 values as a half2 (val)
// This function redistributes elements among threads 
// so that val becomes 2 consecutive fp16 values in column-major order
__device__ void shfl_transpose_warp_uint64(volatile uint64_t &val){
  int col = threadIdx.x/4;
  int row = (threadIdx.x%4)*2;
  uint64_t temp[2];
  temp[0] = __shfl_sync(0xffffffff, val, row*4 + col/2);
  temp[1] = __shfl_sync(0xffffffff, val, (row+1)*4 + col/2);
  if((threadIdx.x/4)%2 == 0){
    val = (((temp[1] >> 32) & 0xFFFF) << 48) |  // B's 32-47 bits
          (((temp[0] >> 32) & 0xFFFF) << 32) |  // A's 32-47 bits
          ((temp[1] & 0xFFFF) << 16)         |  // B's 0-15 bits
          (temp[0] & 0xFFFF);                   // A's 0-15 bits
  } 
  else{
    val = (((temp[1] >> 48) & 0xFFFF) << 48) |  // B's 48-63 bits
          (((temp[0] >> 48) & 0xFFFF) << 32) |  // A's 48-63 bits
          (((temp[1] >> 16) & 0xFFFF) << 16) |  // B's 16-31 bits
          ((temp[0] >> 16) & 0xFFFF);           // A's 16-31 bits
  }
}

std::vector<torch::Tensor> 
f3sCuda1tb1rwScheduledm16n8k8(
    torch::Tensor rowWindowOffset,
    torch::Tensor sortedRowWindows,
    torch::Tensor sparseAToXidx,
    torch::Tensor tcbBitMap,
    int nNodes,
    int embeddingDim,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int nWarpPerBlock){
  int nTcb = sparseAToXidx.size(0)/BLK_N;
  torch::Tensor sddmmResult = torch::zeros({nTcb*BLK_M*BLK_N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)); 
  int nRowWindow = rowWindowOffset.size(0) - 1;
  int paddedLength = nRowWindow * BLK_M; 
  auto output = torch::zeros({paddedLength, embeddingDim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  int sharedSize = BLK_M * embeddingDim * sizeof(half); // Q
  sharedSize += nWarpPerBlock * BLK_M * BLK_N * sizeof(half); // E
  sharedSize += nWarpPerBlock * 2 * BLK_M * sizeof(float); // row_max, row_sum, old_max, old_sum
  sharedSize += BLK_M * embeddingDim * sizeof(float); // O_frag
  // sharedSize += nWarpPerBlock * 2 * BLK_K * BLK_N * sizeof(half); // V double buffer
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

// use m16n8k8 in the spmm phase. 
// This avoids using 16x16 TC blocks and instead uses 16x8 TC blocks.
// The downside is that hmma instructions increase. Overall, not worth it.
__global__ void f3sKernel1tb1rwScheduledk8(
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

  scheduler.next_iter(sortedRowWindows, nRw);
  int niter = (rowWindowOffset[scheduler.targetRw+1] - rowWindowOffset[scheduler.targetRw]
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
  for(int iter = 0; iter < niter; iter++){
    int iterTcbStart = rowWindowOffset[scheduler.targetRw] + iter*blockDim.y;
    // number of 16x8 blocks in S/E being computed in this iteration.
    int nBlock = min(blockDim.y, rowWindowOffset[scheduler.targetRw+1]-iterTcbStart);
    float S_frag[4] = {0.0f};
    int warpTcbId = warpId + iterTcbStart;
    if(warpId < nBlock){
      {//sddmm
        int kOffset = sparseAToXidx[warpTcbId*BLK_N + laneId/4] 
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
          S_frag[i] = (tcbBitMap[warpTcbId*2+i/2] & bitMask) == 0 ? 0.0f : S_frag[i];
        }
      }
      {//save sddmm result
        int offset = warpTcbId*BLK_M*BLK_N + laneId*2;
        for(int j = 0; j < 2; j++){ // 2 8x8 blocks in each 16x8 block
          int sumOffset = j*BLK_N*BLK_N;
          float2 val;
          val.x = S_frag[j*2];
          val.y = S_frag[j*2+1];
          sddmmResult[(offset + sumOffset)/2] = val;
        }
      }
      {//online softmax
        float* maxPtr = reinterpret_cast<float*>(dynShm1tb1rw) 
                        + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2 
                        + laneId/4 + blockDim.y*BLK_M;
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
    if(warpId < nBlock){
      float* sumPtr = reinterpret_cast<float*>(dynShm1tb1rw) 
                      + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2;
      for(int i=0; i<2; i++){
        int offset = i*BLK_M/2 + laneId/4;
        float* maxPtr = sumPtr + blockDim.y*BLK_M + offset;
        float newGlobalMax = maxOld[offset];
        //we have blockDim.y columns reserved for local sum of each warp
        //but only nBlock are used
        for(int j=0; j<nBlock; j++){
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
    //Without warpId == 0, the compiler gets confused and the long scoreboard stall will increase significantly.
    if(warpId == 0 && tid < BLK_M){
      float* sumPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2 + laneId;
      //update r_b
      float rowSum = 0.0f;
      //we have blockDim.y columns reserved for local sum of each warp
      //but only nBlock are used
      for(int j=0; j<nBlock; j++){
        rowSum += sumPtr[j*BLK_M];
      }
      sumOld[laneId] = fmaf(mTilde[laneId], sumOld[laneId], rowSum);
    }
    __syncthreads();
    {//SpMM
      int oOffset_base = (embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + laneId;
      for(int i=warpId; i<embeddingDim/BLK_N; i+=blockDim.y){
        int oOffset = oOffset_base + i*BLK_M*BLK_N;
        float O_frag[4];
        loadOFragShm(O_frag, reinterpret_cast<float*>(dynShm1tb1rw)+oOffset, mTilde+laneId/4);
        for(int j=0; j<nBlock; j++){
          //load V
          int sparseAToXidxOffset = (iterTcbStart+j)*BLK_N + (laneId%4)*2;
          Half2Uint32 B_frag;
          int offset = sparseAToXidx[sparseAToXidxOffset]*embeddingDim+ i*BLK_N + laneId/4;
          B_frag.h2.x = V[offset];
          offset = sparseAToXidx[sparseAToXidxOffset+1]*embeddingDim+ i*BLK_N + laneId/4;
          B_frag.h2.y = V[offset];
          //load E
          volatile uint32_t E_frag[2];
          int eOffset = (embeddingDim+j*BLK_N)*BLK_M/2 + laneId;
          loadEFragShm2(E_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
          HMMA1688(O_frag[0], O_frag[1], O_frag[2], O_frag[3], 
                   E_frag[0], E_frag[1], 
                   B_frag.u32, 
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

// use shfl_transpose_warp_uint64 to access V in row-major order
// doesn't seem to be faster
__global__ void f3sKernel1tb1rwScheduledShflV(
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

  scheduler.next_iter(sortedRowWindows, nRw);
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
    //Without warpId == 0, the compiler gets confused and the long scoreboard stall will increase significantly.
    if(warpId == 0 && tid < BLK_M){
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
      uint32_t *V_uint32 = reinterpret_cast<uint32_t*>(V);
      for(int i=warpId; i<embeddingDim/BLK_N; i+=blockDim.y){
        int oOffset = oOffset_base + i*BLK_M*BLK_N;
        volatile float O_frag[4];
        loadOFragShm(O_frag, reinterpret_cast<float*>(dynShm1tb1rw)+oOffset, mTilde+laneId/4);
        for(int j=0; j<nBlock; j++){
          //load V
          int sparseAToXidxOffset = (iterTcbStart+j)*BLK_M + laneId/4;
          int offset0 = sparseAToXidx[sparseAToXidxOffset]*embeddingDim/2+ i*BLK_N/2 + laneId%4;
          int offset1 = sparseAToXidx[sparseAToXidxOffset+BLK_N]*embeddingDim/2+ i*BLK_N/2 + laneId%4;
          uint64_t val = ((uint64_t)V_uint32[offset1] << 32) | V_uint32[offset0];
          //load E
          volatile uint32_t E_frag[4];
          int eOffset = (embeddingDim+j*BLK_M)*BLK_M/2 + laneId;
          loadEFragShm(E_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
          shfl_transpose_warp_uint64(val);
          HMMA16816(O_frag[0], O_frag[1], O_frag[2], O_frag[3], 
                    E_frag[0], E_frag[1], E_frag[2], E_frag[3], 
                    static_cast<uint32_t>(val & 0xFFFFFFFF), 
                    static_cast<uint32_t>(val >> 32), 
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

__global__ void f3sKernel1tb1rwScheduledDBK(
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
  __shared__ __align__(16) ulonglong2 k_shm_total[8*2*BLK_K*2*BLK_N/8];

  scheduler.next_iter(sortedRowWindows, nRw);
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
        // load first K, each warp's tile contains 4x8 ulonglong2, which is BLK_N*BLK_N/2
        int kShmOffset = warpId*BLK_N*BLK_N/2 + laneId;
        k_shm_total[kShmOffset] = K[kOffset + BLK_K/8];
        for(int i = 0; i < embeddingDim/BLK_K; i+=2) {
          if(i+2 < embeddingDim/BLK_K){
            int kNextShmOffset = ((i+2)/2%2)*8*BLK_N*BLK_N/2 + kShmOffset;
            int shm_ptr = __cvta_generic_to_shared(k_shm_total + kNextShmOffset);
            assert(shm_ptr % 16 == 0);
            asm ("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(shm_ptr), "l"(K + kOffset + (i+2)*BLK_K/8));
          }
          int kCurrentShmOffset = ((i/2)%2)*8*BLK_N*BLK_N/2 + kShmOffset;
          //load K with permuted columns
          ulonglong2 val = k_shm_total[kCurrentShmOffset];
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
          if(i+2 < embeddingDim/BLK_K){
            asm ("cp.async.commit_group;\n"::);
            asm ("cp.async.wait_group 0;\n" ::);
          }
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
    //Without warpId == 0, the compiler gets confused and the long scoreboard stall will increase significantly.
    if(warpId == 0 && tid < BLK_M){
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
          //load E
          uint32_t E_frag[4];
          int eOffset = (embeddingDim+j*BLK_M)*BLK_M/2 + laneId;
          loadEFragShm(E_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
          //load V
          uint32_t B_frag[2];
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

//double buffer for V
__global__ void f3sKernel1tb1rwScheduledDBV(
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
  __shared__ __align__(8) uint64_t V_shm_total[16*BLK_K*BLK_N/4];
  uint64_t* V_shm = V_shm_total + warpId*BLK_K*BLK_N/4;

  scheduler.next_iter(sortedRowWindows, nRw);
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
    //Without warpId == 0, the compiler gets confused and the long scoreboard stall will increase significantly.
    if(warpId == 0 && tid < BLK_M){
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
      // uint64_t* V_shm = dynShm1tb1rw 
      //                   + ((embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + embeddingDim*BLK_M)/2 
      //                   + warpId*BLK_K*BLK_N/4;
      for(int i=warpId; i<embeddingDim/BLK_N; i+=blockDim.y){
        int sparseAToXidxOffset = (iterTcbStart)*BLK_M + laneId/2;
        V_shm[laneId] = reinterpret_cast<uint64_t*>(V)[(sparseAToXidx[sparseAToXidxOffset]*embeddingDim + i*BLK_N)/4 + laneId%2];
        int oOffset = oOffset_base + i*BLK_M*BLK_N;
        volatile float O_frag[4];
        loadOFragShm(O_frag, reinterpret_cast<float*>(dynShm1tb1rw)+oOffset, mTilde+laneId/4);
        for(int j=0; j<nBlock; j++){
          uint32_t E_frag[4];
          //load E
          int eOffset = (embeddingDim+j*BLK_M)*BLK_M/2 + laneId;
          loadEFragShm(E_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
          half* V_shm_half = reinterpret_cast<half*>(V_shm) 
                             + (j%2)*blockDim.y*BLK_K*BLK_N
                             + (laneId%4)*2*BLK_N+laneId/4;
          Half2Uint32 h2U32Converter0;
          h2U32Converter0.h2.x = V_shm_half[0];
          h2U32Converter0.h2.y = V_shm_half[BLK_N];
          Half2Uint32 h2U32Converter1;
          h2U32Converter1.h2.x = V_shm_half[BLK_N*BLK_N];
          h2U32Converter1.h2.y = V_shm_half[BLK_N*BLK_N + BLK_N];
          if(j+1 < nBlock){
            // sparseAToXidxOffset = (iterTcbStart+j+1)*BLK_M + laneId/2;
            // int shm_ptr = __cvta_generic_to_shared(V_shm + ((j+1)%2)*blockDim.y*BLK_K*BLK_N/4 + laneId);
            // uint64_t* V_uint64 = reinterpret_cast<uint64_t*>(V) + (sparseAToXidx[sparseAToXidxOffset]*embeddingDim + i*BLK_N)/4 + laneId%2;
            sparseAToXidxOffset = (iterTcbStart+j)*BLK_M + laneId/4;
            int shm_ptr = __cvta_generic_to_shared(reinterpret_cast<uint32_t*>(V_shm) + ((j+1)%2)*blockDim.y*BLK_K*BLK_N/2 + laneId);
            uint32_t* V_uint32 = reinterpret_cast<uint32_t*>(V) + (sparseAToXidx[sparseAToXidxOffset]*embeddingDim + i*BLK_N)/2 + laneId%4;
            asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(shm_ptr), "l"(V_uint32));
            sparseAToXidxOffset += BLK_N;
            shm_ptr = __cvta_generic_to_shared(reinterpret_cast<uint32_t*>(V_shm) + ((j+1)%2)*blockDim.y*BLK_K*BLK_N/2 + BLK_K*BLK_N/4 + laneId);
            V_uint32 = reinterpret_cast<uint32_t*>(V) + (sparseAToXidx[sparseAToXidxOffset]*embeddingDim + i*BLK_N)/2 + laneId%4;
            asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(shm_ptr), "l"(V_uint32));
            asm ("cp.async.commit_group;\n"::);
          }
          HMMA16816(O_frag[0], O_frag[1], O_frag[2], O_frag[3], 
                    E_frag[0], E_frag[1], E_frag[2], E_frag[3], 
                    h2U32Converter0.u32, h2U32Converter1.u32, 
                    O_frag[0], O_frag[1], O_frag[2], O_frag[3]);
          if(j+1 < nBlock){
            asm ("cp.async.wait_group 0;\n" ::);
          }
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
