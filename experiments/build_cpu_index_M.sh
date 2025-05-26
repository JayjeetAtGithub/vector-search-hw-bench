#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

BINARY_DIR="/workspace/vector-search-hw-bench/src/"
INDEX_DIR="/workspace/vector-search-hw-bench/src/"

build_hnsw_deep() {
  numactl -C 0-23 --membind 0 ${BINARY_DIR}/run_cpu \
      --index-type hnsw \
      --dataset-dir /workspace/dataset/deep \
      --learn-limit ${1} \
      --metric l2 \
      --M ${2} \
      --index-file ${INDEX_DIR}/cpu_hnsw_deep_${1}l_${2}m.faiss
}

build_hnsw_deep 1000000 16
build_hnsw_deep 1000000 32
build_hnsw_deep 1000000 64
build_hnsw_deep 1000000 128
build_hnsw_deep 1000000 256
build_hnsw_deep 1000000 512
build_hnsw_deep 1000000 1024

du -h cpu_hnsw_deep_1000000l_*.faiss
