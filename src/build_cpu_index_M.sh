#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

build_hnsw() {
  ./run_cpu \
      --index-type hnsw \
      --dataset-dir /workspace/dataset/t2i \
      --learn-limit ${1} \
      --metric ip \
      --M ${2} \
      --index-file cpu_hnsw_${1}l_${2}M.faiss
}

# Build IVF index
build_hnsw 1000000 16
build_hnsw 1000000 32
build_hnsw 1000000 64
build_hnsw 1000000 128
build_hnsw 1000000 256
build_hnsw 1000000 512
build_hnsw 1000000 1024

# print the size of the index
du -h cpu_hnsw_1000000l_16M.faiss
du -h cpu_hnsw_1000000l_32M.faiss
du -h cpu_hnsw_1000000l_64M.faiss
du -h cpu_hnsw_1000000l_128M.faiss
du -h cpu_hnsw_1000000l_256M.faiss
du -h cpu_hnsw_1000000l_512M.faiss
du -h cpu_hnsw_1000000l_1024M.faiss
