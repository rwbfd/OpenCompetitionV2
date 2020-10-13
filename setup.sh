#!/bin/bash
python cython/setup install

python setup.py sdist bdist_wheel
pip install dist/open-competition-0.1.tar.gz

