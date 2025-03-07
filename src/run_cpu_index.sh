#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

run_flat() {
    ./run_cpu \
        --index-type flat \
        --dataset-dir /workspace/dataset/t2i \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k 10 \
        --metric ip \
        --skip-build 1 \
        --index-file cpu_flat_${1}l.faiss \
        --calc-recall true
}

run_ivf() {
    ./run_cpu \
        --index-type ivf \
        --dataset-dir /workspace/dataset/t2i \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k 10 \
        --n-probe ${3} \
        --metric ip \
        --skip-build 1 \
        --index-file cpu_ivf_${1}l.faiss \
        --calc-recall true
}

run_hnsw() {
    ./run_cpu \
        --index-type hnsw \
        --dataset-dir /workspace/dataset/t2i \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k 10 \
        --ef ${3} \
        --metric ip \
        --skip-build 1 \
        --index-file cpu_hnsw_${1}l.faiss \
        --calc-recall true
}

run_flat 20000000 1000 
run_flat 20000000 10000
run_flat 20000000 100000

run_ivf  20000000 1000  512
run_ivf  20000000 10000 512
run_ivf  20000000 100000 512

run_hnsw 20000000 1000  512
run_hnsw 20000000 10000 512
run_hnsw 20000000 100000 512
