#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

build_hnsw() {
  ./run_cpu \
      --index-type hnsw \
      --dataset-dir /workspace/dataset/deep \
      --learn-limit ${1} \
      --metric l2 \
      --M ${2} \
      --index-file cpu_hnsw_deep_${1}l_${2}m.faiss
}

build_hnsw 1000000 128
build_hnsw 2000000 128
build_hnsw 4000000 128
build_hnsw 8000000 128
build_hnsw 16000000 128
build_hnsw 32000000 128
build_hnsw 64000000 128
build_hnsw 128000000 128
