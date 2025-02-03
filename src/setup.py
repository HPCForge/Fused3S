from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='TCFMM',
    ext_modules=[
        CUDAExtension('TCFMM', [
            'TCFMM.cpp',
            'TCFMM_kernel.cu',
            'utils.cu'
        ],
        extra_compile_args={
            'nvcc': [
            #   '-maxrregcount=64',
            #   '-O2',
              '-O3',
              '-use_fast_math',
              '-ftz=true',
              '-prec-div=false',
              '-prec-sqrt=false',
              '-lineinfo',
              '-arch=sm_80',
              # '-Xptxas', '-v',
              ]
        }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
