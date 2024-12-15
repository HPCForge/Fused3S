/pub/zitongl5/NCU-2024.3/ncu --set full -f --export dfgnn_f320_pubmed.ncu-rep --kernel-name "regex:f3s_m16n8k16_cuda_kernel|sddmmCooKernel|softMax_SPMM" python dfgnn_softmax_kernel_only.py
/pub/zitongl5/NCU-2024.3/ncu --set full -f --import-source yes --source-folders /pub/zitongl5/TCFMM/src --export f3s_f320_pubmed.ncu-rep --kernel-name "regex:f3s_m16n8k16_cuda_kernel|sddmmCooKernel|softMax_SPMM" python f3s_kernel_only.py

