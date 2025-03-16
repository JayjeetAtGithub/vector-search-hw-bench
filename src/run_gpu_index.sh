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

run_flat 100000 1000
run_flat 100000 10000
run_flat 100000 100000

run_ivf  100000 1000   32
run_ivf  100000 10000  32
run_ivf  100000 100000 32

run_flat 1000000 1000
run_flat 1000000 10000
run_flat 1000000 100000

run_ivf  1000000 1000   48
run_ivf  1000000 10000  48
run_ivf  1000000 100000 48

run_flat 10000000 1000
run_flat 10000000 10000
run_flat 10000000 100000

run_ivf  10000000 1000   64
run_ivf  10000000 10000  64
run_ivf  10000000 100000 64
