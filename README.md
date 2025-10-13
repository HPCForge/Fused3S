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
> + `Dgl/2.4.0` (optional)
> + `PyG/2.6.1` (optional)
> + Nvidia A30/H100 GPU

**Clone this repo and submodules**
  
```shell
git clone --recursive git@github.com:HPCForge/Fused3S.git
```

**Build using Docker image**  

We provide a dockerfile to build the environment needed to run F3S and baseline methods.
To build, clone this repository and its submodules. 
Run the following command in the cloned F3S directory.
```shell
docker build -t fused3s -f dockerfile .
```

**Build from source**
  
Assuming the dependencies are satisfied.
```shell
# To build Fused3S itself
cd src
source build.sh
# To build DF-GNN and FlashSparse as baselines
cd baselines/DF-GNN/
source install.sh
cd baselines/flashSparse/FlashSparse
source compile.sh
```

## How to use Fused3S

```python
import F3S

### Preprocess sparse mask A in CSR format into TC blocks (size BLK_H x BLK_W)
num_row_windows = (A.size + BLK_H - 1) // BLK_H
edgeToColumn = torch.zeros(A.nnz, dtype=torch.int, device='cuda')
edgeToRow = torch.zeros(A.nnz, dtype=torch.int, device='cuda')
blockPartition = torch.zeros(num_row_windows, dtype=torch.int, device='cuda')
indices = torch.IntTensor(A.indices).cuda()
indptr = torch.IntTensor(A.indptr).cuda()
RowWindowOffset, sortedRowWindows, TCblockRowid, _, _,\
SparseAToXindex, TCblockBitMap, _ = F3S.preprocess_gpu(indices, indptr, size, 
                                                       BLK_H, BLK_W, 
                                                       blockPartition, 
                                                       edgeToColumn, 
                                                       edgeToRow)

### Q,K,V should be 2D half precision torch tensor of shape [A.size, embeddingDim]
### nWarpPerBlock is tunable, we recommend to use 8 as a default
time, fusedR = F3S.f3s_1tb1rw_scheduled_permuteV(RowWindowOffset, sortedRowWindows,
                                                 SparseAToXindex, TCblockBitMap, 
                                                 size, Q_half, K_half, V_half,
                                                 nWarpPerBlock)
### Other variantion of Fused3S takes in similar sets of parameters
```

## Tests and reproducibility
**Datasets**

Full graph datasets can be collected by running `scripts/downloadDataset.py`.
Datasets less easy to find is included in `dataset/`

**To profile individual kernels**
```shell
ncu --set full -f --import-source yes --source-folders F3S/src --export f3s_pubmed.ncu-rep --kernel-name "regex:f3sKernel1tb1rwScheduledPermutedQKVScaleQK" python baseline_comp_kernel_only.py -d pubmed -m f3s -a f3s_1tb1rw_scheduled_permuteV
```

**To verify correctness**
```shell
cd scripts/tests
python test_f3s_accuracy.py
```

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
  year={2025},
  doi = {10.1145/3721145.3730430}
}
```
