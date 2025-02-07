#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

run_flat() {
    ./run_faiss_gpu \
        --index-type flat \
        --dataset-dir /workspace/dataset/deep1b \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k 10 \
        --metric ip \
        --index-file gpu_flat_${1}l_${2}q.faiss

    ./run_faiss_gpu \
        --index-type flat \
        --dataset-dir /workspace/dataset/deep1b \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k 10 \
        --metric ip \
        --skip-build 1 \
        --index-file gpu_flat_${1}l_${2}q.faiss \
        --calc-recall true
}

run_ivf() {
    ./run_faiss_gpu \
        --index-type ivf \
        --dataset-dir /workspace/dataset/deep1b \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k 10 \
        --n-probe ${3} \
        --metric ip \
        --index-file gpu_ivf_${1}l_${2}q.faiss

    ./run_faiss_gpu \
        --index-type ivf \
        --dataset-dir /workspace/dataset/deep1b \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k 10 \
        --n-probe ${3} \
        --metric ip \
        --skip-build 1 \
        --index-file gpu_ivf_${1}l_${2}q.faiss \
        --calc-recall true
}

# run flat on 100k learn vectors and 1k search vectors
run_flat 100000 1000

# run ivf on 100k learn vectors and 1k search vectors
run_ivf 100000 1000 128

# run flat on 1M learn vectors and 10k search vectors
run_flat 1000000 10000

# run ivf on 1M learn vectors and 10k search vectors
run_ivf 1000000 10000 256

# run flat on 10M learn vectors and 100k search vectors
run_flat 10000000 100000

# run ivf on 10M learn vectors and 100k search vectors
run_ivf 10000000 100000 512
