from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(
        name='TAD',
        version='1.0',
        packages=find_packages(exclude=('cache', 'output', 'dataset'))
    )
    
