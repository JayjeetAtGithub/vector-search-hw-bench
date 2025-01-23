#!/bin/bash
set -ex

faiss_dir=${1:-thirdparty/faiss}

cd ${faiss_dir}

rm -rf build/
mkdir -p build/
cd build/

sudo apt-get install -y libssl-dev libblas-dev libopenblas-dev liblapack-dev swig python3-numpy

## turn off -DFAISS_OPT_LEVEL=avx2 on grace hopper systems
cmake -DFAISS_ENABLE_GPU=OFF \
      -DFAISS_ENABLE_PYTHON=OFF \
      -DBUILD_SHARED_LIBS=ON \
      -DFAISS_OPT_LEVEL=avx512_spr \
      -DBUILD_TESTING=OFF \
      -DCMAKE_BUILD_TYPE=Release \
     ..

sudo make -j$(nproc)
sudo make install
