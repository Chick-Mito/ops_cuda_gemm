from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

setup(
    name='gemm_ops',
    ext_modules=[
        CUDAExtension(
            name='gemm_ops',
            sources=[
                'src/gemm_kernels.cu',
                'src/gemm_kernels_async.cu',
                'src/gemm_kernels_tc.cu',
                'src/gemm_kernels_tc_async.cu',
                'src/gemm_wrapper.cpp'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-gencode=arch=compute_86,code=sm_86']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
