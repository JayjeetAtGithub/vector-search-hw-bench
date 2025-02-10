#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

learn_limit=100000
search_limit=1000
calc_recall=${1:-false}


run_flat() {
  ./run_faiss_cpu \
    --index-type flat \
    --dataset-dir /workspace/dataset/deep1b \
    --learn-limit ${1} \
    --search-limit ${2} \
    --top-k 10 \
    --ef 256 \
    --metric ip \
    --index-file cpu_flat_${1}l_${2}q.faiss

  perf stat -e fp_arith_inst_retired.512b_packed_single \
            -e fp_arith_inst_retired.256b_packed_single \
            -e fp_arith_inst_retired.128b_packed_single \
            -e fp_arith_inst_retired.vector \
            -e fp_arith_inst_retired.scalar_single \
            -e cache-misses \
            -e instructions \
            -e cpu-cycles \
            -e branch-instructions \
            -e branch-misses \
            ./run_faiss_cpu \
              --index-type flat \
              --dataset-dir /workspace/dataset/deep1b \
              --learn-limit ${1} \
              --search-limit ${2} \
              --top-k 10 \
              --ef 256 \
              --metric ip \
              --skip-build 1 \
              --index-file cpu_flat_${1}l_${2}q.faiss \
              --calc-recall true

}

run_ivf() {
  ./run_faiss_cpu \
      --index-type ivf \
      --dataset-dir /workspace/dataset/deep1b \
      --learn-limit ${1} \
      --search-limit ${2} \
      --top-k 10 \
      --ef 256 \
      --n-probe ${3} \
      --metric ip \
      --index-file cpu_ivf_${1}l_${2}q.faiss

  perf stat -e fp_arith_inst_retired.512b_packed_single \
            -e fp_arith_inst_retired.256b_packed_single \
            -e fp_arith_inst_retired.128b_packed_single \
            -e fp_arith_inst_retired.vector \
            -e fp_arith_inst_retired.scalar_single \
            -e cache-misses \
            -e instructions \
            -e cpu-cycles \
            -e branch-instructions \
            -e branch-misses \
            ./run_faiss_cpu \
              --index-type ivf \
              --dataset-dir /workspace/dataset/deep1b \
              --learn-limit ${1} \
              --search-limit ${2} \
              --top-k 10 \
              --ef 256 \
              --n-probe ${3} \
              --metric ip \
              --skip-build 1 \
              --index-file cpu_ivf_${1}l_${2}q.faiss \
              --calc-recall true
}

run_hnsw() {
  ./run_faiss_cpu \
      --index-type hnsw \
      --dataset-dir /workspace/dataset/deep1b \
      --learn-limit ${learn_limit} \
      --search-limit ${search_limit} \
      --top-k 10 \
      --ef 256 \
      --metric ip \
      --index-file cpu_hnsw_${1}l_${2}q.faiss

  perf stat -e fp_arith_inst_retired.512b_packed_single \
            -e fp_arith_inst_retired.256b_packed_single \
            -e fp_arith_inst_retired.128b_packed_single \
            -e fp_arith_inst_retired.vector \
            -e fp_arith_inst_retired.scalar_single \
            -e cache-misses \
            -e instructions \
            -e cpu-cycles \
            -e branch-instructions \
            -e branch-misses \
            ./run_faiss_cpu \
              --index-type hnsw \
              --dataset-dir /workspace/dataset/deep1b \
              --learn-limit ${1} \
              --search-limit ${2} \
              --top-k 10 \
              --ef 256 \
              --metric ip \
              --skip-build 1 \
              --index-file cpu_hnsw_${1}l_${2}q.faiss \
              --calc-recall true
}

run_flat 1000000 10000 
run_ivf 1000000 10000 256
run_hnsw 1000000 10000
