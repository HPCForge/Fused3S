from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Configure setup
setup(
    name='TCFMM',
    ext_modules=[
        CUDAExtension('TCFMM', [
            'TCFMM.cpp',
            'TCFMM_kernel.cu',
        ],
        extra_compile_args={
            'nvcc': ['-O2', '-lineinfo']
        }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
