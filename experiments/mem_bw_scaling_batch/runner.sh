#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

BINARY_PATH="/workspace/vector-search-hw-bench/src/run_cpu"

numactl -C 0-23 --membind 0 ${BINARY_PATH} \
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
