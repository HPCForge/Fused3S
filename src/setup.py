from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='F3S',
    ext_modules=[
        CUDAExtension('F3S', [
            'F3S.cpp',
            'F3S_kernel.cu',
            'utils.cu'
        ],
        extra_compile_args={
            'nvcc': [
              '-O3',
              '-use_fast_math',
              '-ftz=true',
              '-prec-div=false',
              '-prec-sqrt=false',
              '-lineinfo',
              ]
        }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
