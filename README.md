## Fused3S: Fast Sparse Attention on Tensor Cores
Fused3S is a CUDA kernel library that accelerates sparse attention by fusing Sampled Dense-Dense Matrix Multiplication (SDDMM), Softmax, and Sparse Matrix Multiplication (SpMM) into a single optimized kernel while also leveraging the high throughput of tensor cores. The kernels are optimized for Ampere architecture with ongoing work to exploit new features introduced in Hopper.

## Dependencies
+ **Requirements**: 
> + `CUDA/12.1`
> + `GCC/11.2`
> + `Pytorch/2.4.0`
> + `Dgl/2.4.0`
> + `PyG/2.6.1`
> + Nvidia A30/H100 GPU

## Clone this repo and submodules
```shell
git clone --recursive git@github.com:HPCForge/Fused3S.git
```

## Build using Docker image
We provide a dockerfile to build the environment needed to run F3S and baseline methods.
To build, clone this repository and its submodules. 
Run the following command in the cloned F3S directory.
```shell
docker build -t fused3s -f dockerfile .
```

## Build from source
Assuming the dependencies are satisfied.
```shell
cd src
source build.sh
cd baselines/DF-GNN/
source install.sh
cd baselines/flashSparse/FlashSparse
source compile.sh
```

## Reproduce results in Figure 5
```shell
cd scripts/baseline_comp
python baseline_comp_kernel_only.py -d all -m all -a all --use_event_timer
```

## Reproduce results in Figure 6
```shell
cd scripts/baseline_comp
python baseline_comp_kernel_only.py -d reddit -m f3s -a f3s_1tb1rw --check_sm_active_time
python baseline_comp_kernel_only.py -d reddit -m f3s -a f3s_1tb1rw_scheduled --check_sm_active_time
```

## To profile individual kernel with ncu
```shell
ncu --set full -f --import-source yes --source-folders F3S/src --export f3s_pubmed.ncu-rep --kernel-name "regex:f3sKernel1tb1rwScheduledPermutedQKVScaleQK" python baseline_comp_kernel_only.py -d pubmed -m f3s -a f3s_1tb1rw_scheduled_permuteV
```

## Reproduce results in Figure 7
```shell
cd baselines/graphtransformer
python eval.py
```

## Verifying correctness
```shell
cd scripts/tests
python test_f3s_accuracy.py
```

## Publication
Fused3S is accepted to ICS'25. To cite our work:
```bibtex
@misc{li2025fused3sfastsparseattention,
      title={Fused3S: Fast Sparse Attention on Tensor Cores}, 
      author={Zitong Li and Aparna Chandramowlishwaran},
      year={2025},
      eprint={2505.08098},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2505.08098}, 
}
```
