#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

run_hnsw_deep() {
    pcm-memory 0.5 --csv="pcm_bw_hnsw_deep_${2}s" -- "numactl -C 0-23 --membind 0 ./run_cpu \
        --index-type hnsw \
        --dataset-dir /workspace/dataset/deep \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k ${3} \
        --ef ${4} \
        --metric l2 \
        --skip-build 1 \
        --index-file cpu_hnsw_deep_${1}l_${5}m.faiss \
        --calc-recall false"
}

# batch size = 100, Recall@10 = 95%
run_hnsw_deep 128000000 100 10 650 32
run_hnsw_deep 128000000 500 10 650 32
run_hnsw_deep 128000000 1000 10 650 32
run_hnsw_deep 128000000 2000 10 650 32
run_hnsw_deep 128000000 5000 10 650 32
run_hnsw_deep 128000000 10000 10 650 32
