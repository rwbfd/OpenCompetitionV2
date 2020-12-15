import setuptools
from distutils.core import Extension
from Cython.Build import cythonize
import numpy

compile_flags = ['-std=c++11']

module = Extension('ind_cols',
                   ['cython/ind_cols.pyx'],
                   language='c++',
                   include_dirs=[numpy.get_include()],
                   extra_compile_args=compile_flags)


setuptools.setup(
    name="open-competition",
    version="0.1",
    author="Ran Wang",
    author_email="ran.wang.math@gmail.com",
    description="A package for empirical data science competition.",
    packages=setuptools.find_packages(),
    ext_modules=cythonize(module),
    python_requires='>=3.6',
)
