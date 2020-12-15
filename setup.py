import setuptools
from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy

compile_flags = ['-std=c++11']

module = Extension('cython/ind_cols',
                   ['cython/ind_cols.pyx'],
                   language='c++',
                   include_dirs=[numpy.get_include()],
                   extra_compile_args=compile_flags)

setup(
    name='cython_test',
    ext_modules=cythonize(module)
)


setuptools.setup(
    name="open-competition",
    version="0.1",
    author="Ran Wang",
    author_email="ran.wang.math@gmail.com",
    description="A package for empirical data science competition.",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
