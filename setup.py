import setuptools
from Cython.Build import cythonize

setuptools.setup(
    name="open-competition",
    version="0.1",
    author="Ran Wang",
    author_email="ran.wang.math@gmail.com",
    description="A package for empirical data science competition.",
    ext_modules=cythonize("cython/ind_cols.pyx"),
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
