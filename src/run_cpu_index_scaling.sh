#!/bin/bash
set -ex

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

run_hnsw() {
    numactl -C ${5} --membind 0 ./run_cpu \
        --index-type hnsw \
        --dataset-dir /workspace/dataset/deep \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k ${3} \
        --ef ${4} \
        --metric l2 \
        --skip-build 1 \
        --index-file cpu_hnsw_${1}l_${6}m.faiss \
        --calc-recall true \
        --gt-file gt_${1}l_${2}s_${3}k.bin
}

run_hnsw 128000000 10000 10 650 0-0 32
run_hnsw 128000000 10000 10 650 0-1 32
run_hnsw 128000000 10000 10 650 0-3 32
run_hnsw 128000000 10000 10 650 0-7 32
run_hnsw 128000000 10000 10 650 0-15 32 
run_hnsw 128000000 10000 10 650 0-23 32
