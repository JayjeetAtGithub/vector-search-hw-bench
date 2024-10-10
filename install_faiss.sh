#!/bin/bash
set -ex

faiss_dir=$1

cd ${faiss_dir}

rm -rf build/
mkdir -p build/
cd build/

sudo apt-get install -y libssl-dev libblas-dev libopenblas-dev liblapack-dev swig python3-numpy

cmake -DFAISS_ENABLE_GPU=ON \
      -DFAISS_ENABLE_PYTHON=OFF \
      -DBUILD_SHARED_LIBS=ON \
      -DFAISS_OPT_LEVEL=avx2 \
      -DBUILD_TESTING=OFF \
      -DCMAKE_BUILD_TYPE=Release \
     ..

sudo make -j$(nproc)
sudo make install
