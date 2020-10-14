#!/bin/bash
cd cython
python setup.py install
cd ..

python setup.py sdist bdist_wheel
pip install dist/open-competition-0.1.tar.gz

