#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

run_hnsw() {
    numactl -C ${6} --membind 0 ./run_cpu \
        --index-type hnsw \
        --dataset-dir /workspace/dataset/deep \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k ${3} \
        --ef ${4} \
        --metric l2 \
        --skip-build 1 \
        --index-file cpu_hnsw_deep_${1}l_${5}m.faiss \
        --calc-recall false
}

run_hnsw 128000000 10000 10 650 32 0-0
run_hnsw 128000000 10000 10 650 32 0-1
run_hnsw 128000000 10000 10 650 32 0-3
run_hnsw 128000000 10000 10 650 32 0-7
run_hnsw 128000000 10000 10 650 32 0-15
run_hnsw 128000000 10000 10 650 32 0-23
