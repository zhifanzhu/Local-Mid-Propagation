import os
import platform
import time
from setuptools import Extension, dist, find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def make_cuda_ext(name, module, sources):
    return CUDAExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })

if __name__ == '__main__':
    setup(
        name='psroi_pool_build',
        package_data={'.': ['psroi_pool_build_output.so']},
        ext_modules=[
            make_cuda_ext(
                name='psroi_pool_cuda',
                module='.',
                sources=['src/psroi_pool_cuda.cpp', 'src/psroi_pool_kernel.cu']),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False,
        verbose=True)
