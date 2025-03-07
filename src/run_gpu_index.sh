#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

run_flat() {
    ./run_gpu \
        --index-type flat \
        --dataset-dir /workspace/dataset/t2i \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k 10 \
        --metric ip \
        --skip-build 1 \
        --index-file gpu_flat_${1}l.faiss \
        --calc-recall true
}

run_ivf() {
    ./run_gpu \
        --index-type ivf \
        --dataset-dir /workspace/dataset/t2i \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k 10 \
        --n-probe ${3} \
        --metric ip \
        --skip-build 1 \
        --index-file gpu_ivf_${1}l.faiss \
        --calc-recall true
}

run_flat 40000000   1000
run_flat 40000000   10000
run_flat 40000000   100000

run_ivf 40000000  1000  1024
run_ivf 40000000  10000 1024
run_ivf 40000000  100000 1024
