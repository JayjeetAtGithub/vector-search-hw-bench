#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

build_flat() {
  ./run_cpu \
    --index-type flat \
    --dataset-dir /workspace/dataset/t2i \
    --learn-limit ${1} \
    --metric ip \
    --index-file cpu_flat_${1}l.faiss
}

build_ivf() {
  ./run_cpu \
      --index-type ivf \
      --dataset-dir /workspace/dataset/t2i \
      --learn-limit ${1} \
      --metric ip \
      --index-file cpu_ivf_${1}l.faiss
}

build_hnsw() {
  ./run_cpu \
      --index-type hnsw \
      --dataset-dir /workspace/dataset/t2i \
      --learn-limit ${1} \
      --metric ip \
      --index-file cpu_hnsw_${1}l.faiss
}


# Build IVF index
build_hnsw 1000000
build_hnsw 2000000
build_hnsw 4000000
build_hnsw 8000000
build_hnsw 16000000
build_hnsw 32000000
