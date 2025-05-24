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

build_hnsw 1000000 16
build_hnsw 1000000 32
build_hnsw 1000000 64
build_hnsw 1000000 128
build_hnsw 1000000 256
build_hnsw 1000000 512
build_hnsw 1000000 1024

# print the size of the index
du -h cpu_hnsw_1000000l_16m.faiss
du -h cpu_hnsw_1000000l_32m.faiss
du -h cpu_hnsw_1000000l_64m.faiss
du -h cpu_hnsw_1000000l_128m.faiss
du -h cpu_hnsw_1000000l_256m.faiss
du -h cpu_hnsw_1000000l_512m.faiss
du -h cpu_hnsw_1000000l_1024m.faiss
