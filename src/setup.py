from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

include_path = '/pub/zitongl5/cccl/install/include'
setup(
    name='TCFMM',
    ext_modules=[
        CUDAExtension('TCFMM', [
            'TCFMM.cpp',
            'TCFMM_kernel.cu'
        ],
        include_dirs=[include_path],
        extra_compile_args={
            'nvcc': [
            #   '-maxrregcount=32',
            #   '-O2',
              '-O3',
              '-use_fast_math',
              '-ftz=true',
              '-lineinfo',
              '-I' + include_path]
        }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
