embedding_dim=128
nWarpPerBlock=-1

/pub/zitongl5/NCU-2024.3/ncu --set full -f --import-source yes --source-folders /pub/zitongl5/TCFMM/src --export f3s_comp_f${embedding_dim}_w${nWarpPerBlock}_pubmed.ncu-rep --kernel-name "regex:f3sKernel1tb1tcb|sddmmKernel1tbnrw|f3sKernel1tb1rw" python f3s_kernel_only.py -emb ${embedding_dim} -nw ${nWarpPerBlock} -d pubmed


