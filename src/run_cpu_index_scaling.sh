#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

run_hnsw() {
    numactl -C ${5} --membind 0 ./run_cpu \
        --index-type hnsw \
        --dataset-dir /workspace/dataset/t2i \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k ${3} \
        --ef ${4} \
        --metric ip \
        --skip-build 1 \
        --index-file cpu_hnsw_${1}l.faiss \
        --calc-recall true \
        --gt-file gt_${1}l_${2}s_${3}k.bin
}

run_hnsw 1000000  100000 10 250 0-0
run_hnsw 1000000  100000 10 250 0-1
run_hnsw 1000000  100000 10 250 0-3
run_hnsw 1000000  100000 10 250 0-7
run_hnsw 1000000  100000 10 250 0-15
run_hnsw 1000000  100000 10 250 0-23

run_hnsw 32000000 100000 10 2250 0-0
run_hnsw 32000000 100000 10 2250 0-1
run_hnsw 32000000 100000 10 2250 0-3
run_hnsw 32000000 100000 10 2250 0-7
run_hnsw 32000000 100000 10 2250 0-15
run_hnsw 32000000 100000 10 2250 0-23
