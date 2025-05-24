#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

build_hnsw_deep() {
  ./run_cpu \
      --index-type hnsw \
      --dataset-dir /workspace/dataset/deep \
      --learn-limit ${1} \
      --metric l2 \
      --M ${2} \
      --index-file cpu_hnsw_deep_${1}l_${2}m.faiss
}

build_hnsw_t2i() {
  ./run_cpu \
      --index-type hnsw \
      --dataset-dir /workspace/dataset/t2i \
      --learn-limit ${1} \
      --metric ip \
      --M ${2} \
      --index-file cpu_hnsw_t2i_${1}l_${2}m.faiss
}

build_hnsw_deep 1000000 128
build_hnsw_deep 2000000 128
build_hnsw_deep 4000000 128
build_hnsw_deep 8000000 128
build_hnsw_deep 16000000 128
build_hnsw_deep 32000000 128
build_hnsw_deep 64000000 128
build_hnsw_deep 128000000 128

build_hnsw_t2i 1000000 128
build_hnsw_t2i 2000000 128
build_hnsw_t2i 4000000 128
build_hnsw_t2i 8000000 128
build_hnsw_t2i 16000000 128
build_hnsw_t2i 32000000 128
build_hnsw_t2i 64000000 128
build_hnsw_t2i 128000000 128
