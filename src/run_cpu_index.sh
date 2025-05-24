#!/bin/bash
set -ex

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

run_hnsw() {
    ./run_cpu \
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

# Batch size = 100, Recall@10 = 90-91%
run_hnsw 1000000  100 10 450
run_hnsw 2000000  100 10 450
run_hnsw 4000000  100 10 850
run_hnsw 8000000  100 10 1250
run_hnsw 16000000 100 10 1650
run_hnsw 32000000 100 10 2450

# Batch size = 1000, Recall@10 = 90-91%
run_hnsw 1000000  1000 10 250
run_hnsw 2000000  1000 10 450
run_hnsw 4000000  1000 10 850
run_hnsw 8000000  1000 10 1250
run_hnsw 16000000 1000 10 1650
run_hnsw 32000000 1000 10 2250

# Batch size = 10000, Recall@10 = 90-91%
run_hnsw 1000000  10000 10 250
run_hnsw 2000000  10000 10 450
run_hnsw 4000000  10000 10 850
run_hnsw 8000000  10000 10 1250
run_hnsw 16000000 10000 10 1650
run_hnsw 32000000 10000 10 2250

# Batch size = 100000, Recall@10 = 90-91%
run_hnsw 1000000  100000 10 250
run_hnsw 2000000  100000 10 450
run_hnsw 4000000  100000 10 850
run_hnsw 8000000  100000 10 1250
run_hnsw 16000000 100000 10 1650
run_hnsw 32000000 100000 10 2250
