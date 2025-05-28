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

# Build flat index
build_flat 1000000
build_flat 2000000
build_flat 4000000
build_flat 8000000
build_flat 16000000
build_flat 32000000

# Build IVF index
build_ivf 1000000
build_ivf 2000000
build_ivf 4000000
build_ivf 8000000
build_ivf 16000000
build_ivf 32000000
