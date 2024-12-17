from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == '__main__':
    setup(
        name='TAD',
        version='1.0',
        packages=find_packages(exclude=('cache', 'output', 'dataset')),
        cmdclass={
            'build_ext': BuildExtension
        }
    )
    