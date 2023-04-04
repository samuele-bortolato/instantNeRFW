from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='spc',
    ext_modules=[
        CUDAExtension('spc', [
            'spc.cpp',
            'spc_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })