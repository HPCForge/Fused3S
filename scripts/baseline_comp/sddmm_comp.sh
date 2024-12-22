featureDim=128
nWarpPerBlock=4

/pub/zitongl5/NCU-2024.3/ncu --set full -f --export sddmm_comp_f${featureDim}_w${nWarpPerBlock}_reddit.ncu-rep --kernel-name "regex:f3s_m16n8k16_cuda_kernel|sddmm_kernel_1tbnrw|sddmm_kernel_1tb1rw|sddmmCooKernel" python f3s_kernel_only.py -emb ${featureDim} -nw ${nWarpPerBlock}


