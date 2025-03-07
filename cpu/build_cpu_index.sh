#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

build_flat() {
  ./run_cpu \
    --index-type flat \
    --dataset-dir /workspace/dataset/t2i \
    --learn-limit ${1} \
    --search-limit ${2} \
    --top-k 10 \
    --metric ip \
    --index-file cpu_flat_${1}l_${2}q.faiss
}

build_ivf() {
  ./run_cpu \
      --index-type ivf \
      --dataset-dir /workspace/dataset/t2i \
      --learn-limit ${1} \
      --search-limit ${2} \
      --metric ip \
      --index-file cpu_ivf_${1}l_${2}q.faiss
}

build_hnsw() {
  ./run_cpu \
      --index-type hnsw \
      --dataset-dir /workspace/dataset/t2i \
      --learn-limit ${1} \
      --search-limit ${2} \
      --metric ip \
      --index-file cpu_hnsw_${1}l_${2}q.faiss
}

build_flat 100000   10000 
build_flat 1000000  10000
build_flat 10000000 10000

build_ivf  100000   10000
build_ivf  1000000  10000
build_ivf  10000000 10000

build_hnsw 100000   10000
build_hnsw 1000000  10000
build_hnsw 10000000 10000
