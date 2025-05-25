#!/bin/bash
set -ex

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

run_hnsw() {
    ./run_cpu \
        --index-type hnsw \
        --dataset-dir /workspace/dataset/deep \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k ${3} \
        --ef ${4} \
        --metric l2 \
        --skip-build 1 \
        --index-file cpu_hnsw_deep_${1}l_${5}m.faiss \
        --calc-recall true \
        --gt-file gt_${1}l_${2}s_${3}k.bin
}

# Batch size = 10, Recall@10 = 95%
run_hnsw 1000000   10 10 100 32
run_hnsw 2000000   10 10 40  32
run_hnsw 4000000   10 10 75  32
run_hnsw 8000000   10 10 120 32
run_hnsw 16000000  10 10 90  32
run_hnsw 32000000  10 10 320 32
run_hnsw 64000000  10 10 350 32
run_hnsw 128000000 10 10 370 32

# Batch size = 100, Recall@10 = 95%
run_hnsw 1000000   100 10 110  32
run_hnsw 2000000   100 10 115  32
run_hnsw 4000000   100 10 200  32
run_hnsw 8000000   100 10 250 32
run_hnsw 16000000  100 10 320 32
run_hnsw 32000000  100 10 400 32
run_hnsw 64000000  100 10 550 32
run_hnsw 128000000 100 10 650 32

# Batch size = 1000, Recall@10 = 95%
run_hnsw 1000000   1000 10 110  32
run_hnsw 2000000   1000 10 115  32
run_hnsw 4000000   1000 10 200  32
run_hnsw 8000000   1000 10 250 32
run_hnsw 16000000  1000 10 320 32
run_hnsw 32000000  1000 10 400 32
run_hnsw 64000000  1000 10 550 32
run_hnsw 128000000 1000 10 650 32

# Batch size = 10000, Recall@10 = 95%
run_hnsw 1000000   10000 10 110  32
run_hnsw 2000000   10000 10 115  32
run_hnsw 4000000   10000 10 200  32
run_hnsw 8000000   10000 10 250 32
run_hnsw 16000000  10000 10 320 32
run_hnsw 32000000  10000 10 400 32
run_hnsw 64000000  10000 10 550 32
run_hnsw 128000000 10000 10 650 32
