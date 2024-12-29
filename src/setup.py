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
              '-maxrregcount=32',
            #   '-O2',
              '-O3',
              '-use_fast_math',
              '-ftz=true',
              '-lineinfo']
        }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
