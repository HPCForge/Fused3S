## Fused3S

This repository provides the official implementation of Fused3S from the following paper.

Fused3S: Fast Sparse Attention on Tensor Cores  
Zitong Li, Aparna Chandramowlishwaran  
Paper: https://dl.acm.org/doi/full/10.1145/3721145.3730430

Sparse attention forward pass on H100 for single graph and batched graphs datasets.
<img width="7089" height="1661" alt="speedup_full_GH200" src="https://github.com/user-attachments/assets/cf4fd105-be0d-49c2-bfb7-413b87640795" />
<img width="7113" height="1661" alt="speedup_batched_GH200" src="https://github.com/user-attachments/assets/20d1bd13-3fc6-4129-b5da-66936a2f0e88" />

The kernels are optimized for Ampere architecture with ongoing work to exploit new features introduced in Hopper.

## Installation

**Dependencies**
> + `CUDA/12.1`
> + `GCC/11.2`
> + `Pytorch/2.4.0`
> + `Dgl/2.4.0`
> + `PyG/2.6.1`
> + Nvidia A30/H100 GPU

**Clone this repo and submodules**
  
```shell
git clone --recursive git@github.com:HPCForge/Fused3S.git
```

**To build using Docker image**  

We provide a dockerfile to build the environment needed to run F3S and baseline methods.
To build, clone this repository and its submodules. 
Run the following command in the cloned F3S directory.
```shell
docker build -t fused3s -f dockerfile .
```

**To build from source**
  
Assuming the dependencies are satisfied.
```shell
cd src
source build.sh
cd baselines/DF-GNN/
source install.sh
cd baselines/flashSparse/FlashSparse
source compile.sh
```

**To profile individual kernels**
```shell
ncu --set full -f --import-source yes --source-folders F3S/src --export f3s_pubmed.ncu-rep --kernel-name "regex:f3sKernel1tb1rwScheduledPermutedQKVScaleQK" python baseline_comp_kernel_only.py -d pubmed -m f3s -a f3s_1tb1rw_scheduled_permuteV
```

**To verify correctness**
```shell
cd scripts/tests
python test_f3s_accuracy.py
```

## Tests and reproducibility
**Reproduce Figure 5 results**
```shell
cd scripts/baseline_comp
python baseline_comp_kernel_only.py -d all -m all -a all --use_event_timer
```

**Reproduce Figure 6 results**
```shell
cd scripts/baseline_comp
python baseline_comp_kernel_only.py -d reddit -m f3s -a f3s_1tb1rw --check_sm_active_time
python baseline_comp_kernel_only.py -d reddit -m f3s -a f3s_1tb1rw_scheduled --check_sm_active_time
```

**Reproduce Figure 7 results**
```shell
cd baselines/graphtransformer
python eval.py
```

## Citation
If you have found this codebase useful in your research, please cite:
```bibtex
@inproceedings{li2025fused3s,
  title={Fused3S: Fast Sparse Attention on Tensor Cores},
  author={Li, Zitong and Chandramowlishwaran, Aparna},
  booktitle={Proceedings of the 39th ACM International Conference on Supercomputing},
  pages={104--118},
  year={2025}
}
```
