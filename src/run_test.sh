#!/bin/bash
set -ex

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

BINARY_DIR="/workspace/vector-search-hw-bench/src/"
INDEX_DIR="/workspace/vector-search-hw-bench/src/"

run_exp() {
    numactl -C ${1} --membind 0 ${BINARY_DIR}/run_cpu_profile \
            --index-type hnsw \
            --dataset-dir /workspace/dataset/deep \
            --learn-limit 256000000 \
            --search-limit 10000 \
            --top-k 10 \
            --ef 840 \
            --metric l2 \
            --index-file ${INDEX_DIR}/cpu_hnsw_deep_256000000l_32m.faiss
}

run_exp_profile() {
    numactl -C ${1} --membind 0 ${BINARY_DIR}/run_cpu_profile \
            --index-type hnsw \
            --dataset-dir /workspace/dataset/deep \
            --learn-limit 256000000 \
            --search-limit 10000 \
            --top-k 10 \
            --ef 840 \
            --metric l2 \
            --index-file ${INDEX_DIR}/cpu_hnsw_deep_256000000l_32m.faiss \
            --out cpu_hnsw_deep_256M_${1}.perf-stat
}

run_exp 0-0
run_exp 0-1
run_exp 0-3
run_exp 0-7
run_exp 0-15
run_exp 0-31
run_exp 0-63
run_exp 0-71

run_exp_profile 0-0
run_exp_profile 0-1
run_exp_profile 0-3
run_exp_profile 0-7
run_exp_profile 0-15
run_exp_profile 0-31
run_exp_profile 0-63
run_exp_profile 0-71
