#! /bin/bash
mkdir build
cd build || exit
echo "Installing R"
conda install r-base r-essentials

echo "Install cmake to compile"
sudo apt-get install -y openssl libssl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.19.0-rc2/cmake-3.19.0-rc2.tar.gz
echo "Installing packages from pip"
cd ..
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html -i https://mirrors.aliyun.com/pypi/simple
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
