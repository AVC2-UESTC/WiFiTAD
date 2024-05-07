from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == '__main__':
    setup(
        name='TALFi',
        version='1.0',
        packages=find_packages(exclude=('models', 'output', 'datasets')),
        cmdclass={
            'build_ext': BuildExtension
        }
    )
    