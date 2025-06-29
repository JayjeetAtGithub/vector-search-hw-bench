#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

BINARY_DIR="/workspace/vector-search-hw-bench/src/"
INDEX_DIR="/workspace/vector-search-hw-bench/src/"

run_hnsw() {
    numactl -C 0-23 --membind 0 ${BINARY_DIR}/run_cpu \
        --index-type hnsw \
        --dataset-dir /workspace/dataset/deep \
        --learn-limit ${1} \
        --search-limit 10000 \
        --batch-size ${2} \
        --top-k 10 \
        --ef ${3} \
        --metric l2 \
        --skip-build 1 \
        --index-file ${INDEX_DIR}/cpu_hnsw_deep_${1}l_${4}m.faiss \
        --calc-recall false
}

# Batch size = 1, Recall@10 = 95%
run_hnsw 1000000   1 110 32
run_hnsw 2000000   1 115 32
run_hnsw 4000000   1 200 32
run_hnsw 8000000   1 250 32
run_hnsw 16000000  1 320 32
run_hnsw 32000000  1 400 32
run_hnsw 64000000  1 550 32
run_hnsw 128000000 1 650 32

# Batch size = 10, Recall@10 = 95%
run_hnsw 1000000   10 110  32
run_hnsw 2000000   10 115  32
run_hnsw 4000000   10 200  32
run_hnsw 8000000   10 250 32
run_hnsw 16000000  10 320 32
run_hnsw 32000000  10 400 32
run_hnsw 64000000  10 550 32
run_hnsw 128000000 10 650 32

# batch size = 100, Recall@10 = 95%
run_hnsw 1000000   100 110 32
run_hnsw 2000000   100 115 32
run_hnsw 4000000   100 200 32
run_hnsw 8000000   100 250 32
run_hnsw 16000000  100 320 32
run_hnsw 32000000  100 400 32
run_hnsw 64000000  100 550 32
run_hnsw 128000000 100 650 32

# Batch size = 1000, Recall@10 = 95%
run_hnsw 1000000   1000 110 32
run_hnsw 2000000   1000 115 32
run_hnsw 4000000   1000 200 32
run_hnsw 8000000   1000 250 32
run_hnsw 16000000  1000 320 32
run_hnsw 32000000  1000 400 32
run_hnsw 64000000  1000 550 32
run_hnsw 128000000 1000 650 32

# Batch size = 10000, Recall@10 = 95%
run_hnsw 1000000   10000 110 32
run_hnsw 2000000   10000 115 32
run_hnsw 4000000   10000 200 32
run_hnsw 8000000   10000 250 32
run_hnsw 16000000  10000 320 32
run_hnsw 32000000  10000 400 32
run_hnsw 64000000  10000 550 32
run_hnsw 128000000 10000 650 32
