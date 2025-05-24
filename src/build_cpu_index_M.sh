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

build_hnsw_deep 1000000 16
build_hnsw_deep 1000000 32
build_hnsw_deep 1000000 64
build_hnsw_deep 1000000 128
build_hnsw_deep 1000000 256
build_hnsw_deep 1000000 512
build_hnsw_deep 1000000 1024

# build_hnsw_t2i 1000000 16
# build_hnsw_t2i 1000000 32
# build_hnsw_t2i 1000000 64
# build_hnsw_t2i 1000000 128
# build_hnsw_t2i 1000000 256
# build_hnsw_t2i 1000000 512
# build_hnsw_t2i 1000000 1024

# print the size of the index
du -h cpu_hnsw_deep_1000000l_*.faiss
# du -h cpu_hnsw_t2i_1000000l_*.faiss
