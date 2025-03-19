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

run_flat 100000 10
run_flat 100000 100
run_flat 100000 1000
run_flat 100000 10000

run_ivf  100000 10    32
run_ivf  100000 100   32
run_ivf  100000 1000  32
run_ivf  100000 10000 32
 
run_hnsw 100000 10    32
run_hnsw 100000 100   32
run_hnsw 100000 1000  32
run_hnsw 100000 10000 32

run_flat 1000000 10
run_flat 1000000 100
run_flat 1000000 1000
run_flat 1000000 10000

run_ivf  1000000 10    48
run_ivf  1000000 100   48
run_ivf  1000000 1000  48
run_ivf  1000000 10000 48

run_hnsw 1000000 10    96
run_hnsw 1000000 100   96
run_hnsw 1000000 1000  96
run_hnsw 1000000 10000 96

run_flat 10000000 10
run_flat 10000000 100
run_flat 10000000 1000
run_flat 10000000 10000

run_ivf  10000000 10    64
run_ivf  10000000 100   64
run_ivf  10000000 1000  64
run_ivf  10000000 10000 64

run_hnsw 10000000 10    512
run_hnsw 10000000 100   512
run_hnsw 10000000 1000  512
run_hnsw 10000000 10000 512
