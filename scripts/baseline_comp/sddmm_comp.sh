featureDim=128
nWarpPerBlock=8

/pub/zitongl5/NCU-2024.3/ncu --set full -f --import-source yes --source-folders /pub/zitongl5/TCFMM/src --export sddmm_comp_f${featureDim}_w${nWarpPerBlock}_1tb1rwshm128b_reddit.ncu-rep --kernel-name "regex:f3s_m16n8k16_cuda_kernel|sddmm_kernel_1tbnrw|sddmm_kernel_1tb1rw|sddmmCooKernel" python f3s_kernel_only.py -emb ${featureDim} -nw ${nWarpPerBlock}


