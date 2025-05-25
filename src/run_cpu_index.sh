#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

run_hnsw() {
    numactl -C 0-23 --membind 0 ./run_cpu \
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

# batch size = 100, Recall@10 = 95%
run_hnsw 1000000   100 10 110  32
run_hnsw 2000000   100 10 115  32
run_hnsw 4000000   100 10 200  32
run_hnsw 8000000   100 10 250 32
run_hnsw 16000000  100 10 320 32
run_hnsw 32000000  100 10 400 32
run_hnsw 64000000  100 10 550 32
run_hnsw 128000000 100 10 650 32

# Batch size = 500, Recall@10 = 95%
run_hnsw 1000000   500 10 110  32
run_hnsw 2000000   500 10 115  32
run_hnsw 4000000   500 10 200  32
run_hnsw 8000000   500 10 250 32
run_hnsw 16000000  500 10 320 32
run_hnsw 32000000  500 10 400 32
run_hnsw 64000000  500 10 550 32
run_hnsw 128000000 500 10 650 32

# Batch size = 1000, Recall@10 = 95%
run_hnsw 1000000   1000 10 110  32
run_hnsw 2000000   1000 10 115  32
run_hnsw 4000000   1000 10 200  32
run_hnsw 8000000   1000 10 250 32
run_hnsw 16000000  1000 10 320 32
run_hnsw 32000000  1000 10 400 32
run_hnsw 64000000  1000 10 550 32
run_hnsw 128000000 1000 10 650 32

# Batch size = 2000, Recall@10 = 95%
run_hnsw 1000000   2000 10 110  32
run_hnsw 2000000   2000 10 115  32
run_hnsw 4000000   2000 10 200  32
run_hnsw 8000000   2000 10 250 32
run_hnsw 16000000  2000 10 320 32
run_hnsw 32000000  2000 10 400 32
run_hnsw 64000000  2000 10 550 32
run_hnsw 128000000 2000 10 650 32

# Batch size = 5000, Recall@10 = 95%
run_hnsw 1000000   5000 10 110  32
run_hnsw 2000000   5000 10 115  32
run_hnsw 4000000   5000 10 200  32
run_hnsw 8000000   5000 10 250 32
run_hnsw 16000000  5000 10 320 32
run_hnsw 32000000  5000 10 400 32
run_hnsw 64000000  5000 10 550 32
run_hnsw 128000000 5000 10 650 32

# Batch size = 10000, Recall@10 = 95%
run_hnsw 1000000   10000 10 110  32
run_hnsw 2000000   10000 10 115  32
run_hnsw 4000000   10000 10 200  32
run_hnsw 8000000   10000 10 250 32
run_hnsw 16000000  10000 10 320 32
run_hnsw 32000000  10000 10 400 32
run_hnsw 64000000  10000 10 550 32
run_hnsw 128000000 10000 10 650 32
