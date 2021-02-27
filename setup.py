import setuptools
from setuptools import find_packages, setup

import numpy
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension

version_file = 'reid/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def get_extensions():
    extensions = []

    ext_rank = setuptools.Extension(
        name='reid.core.evaluation._rank',
        sources=['./reid/core/evaluation/rank_cylib/rank.pyx'],
        include_dirs=[numpy.get_include()])
    extensions.extend(cythonize(ext_rank))

    return extensions


if __name__ == '__main__':
    setup(
        name='reid',
        version=get_version(),
        packages=find_packages(exclude=('configs', 'tools')),
        ext_modules=get_extensions(),
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
