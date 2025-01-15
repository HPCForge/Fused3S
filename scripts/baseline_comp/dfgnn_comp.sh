# for format in csr_gm softmax_gm tiling csr; do
#   ncu --set full -f --export dfgnn_${format}_f128_reddit.ncu-rep --kernel-name "regex:softMax_SPMM_global_memory|sddmmCooKernel|fused_gt_csr_global_memory|fused_gt_tiling" python dfgnn_super_node_kernel_only.py --format ${format} --embedding_dim 128 --dataset reddit
# done

for format in csr softmax hyper; do
  ncu --set full -f --export dfgnn_${format}_f128_pubmed.ncu-rep --kernel-name "regex:softMax_SPMM|sddmmCooKernel|fused_gt_csr|fused_gt_hyper_inference_vec4|fused_gt_hyper_inference|fused_gt_hyper_inference_small_f" python full_graph_kernel_only_comp.py --format ${format} --embedding_dim 128 --dataset pubmed
done
