#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

BINARY_DIR="/workspace/vector-search-hw-bench/src/"
INDEX_DIR="/workspace/vector-search-hw-bench/src/"

build_flat_deep() {
  ${BINARY_DIR}/run_gpu \
      --index-type flat \
      --dataset-dir /workspace/dataset/deep \
      --learn-limit ${1} \
      --metric l2 \
      --index-file ${INDEX_DIR}/gpu_flat_deep_${1}l_${2}m.faiss
}

build_ivf_deep() {
  ${BINARY_DIR}/run_gpu \
      --index-type ivf \
      --dataset-dir /workspace/dataset/deep \
      --learn-limit ${1} \
      --metric l2 \
      --index-file ${INDEX_DIR}/gpu_ivf_deep_${1}l_${2}m.faiss
}

build_flat_deep 1000000 
build_flat_deep 2000000 
build_flat_deep 4000000 
build_flat_deep 8000000 
build_flat_deep 16000000 
build_flat_deep 32000000

build_ivf_deep 1000000
build_ivf_deep 2000000 
build_ivf_deep 4000000 
build_ivf_deep 8000000 
build_ivf_deep 16000000 
build_ivf_deep 32000000
