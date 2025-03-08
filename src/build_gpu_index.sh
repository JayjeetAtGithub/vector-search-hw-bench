#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

build_flat() {
    ./run_gpu \
        --index-type flat \
        --dataset-dir /workspace/dataset/t2i \
        --learn-limit ${1} \
        --metric ip \
        --index-file gpu_flat_${1}l.faiss
}

build_ivf() {
    ./run_gpu \
        --index-type ivf \
        --dataset-dir /workspace/dataset/t2i \
        --learn-limit ${1} \
        --metric ip \
        --index-file gpu_ivf_${1}l.faiss
}

build_flat 100000
build_ivf  100000

build_flat 1000000
build_ivf  1000000

build_flat 10000000
build_ivf  10000000

build_flat 20000000
build_ivf  20000000

build_flat 40000000
build_ivf  40000000

build_flat 50000000
build_ivf  50000000
