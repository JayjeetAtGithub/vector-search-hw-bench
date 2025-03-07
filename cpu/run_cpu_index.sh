#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

run_flat() {
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
            ./run_cpu \
              --index-type flat \
              --dataset-dir /workspace/dataset/t2i \
              --learn-limit ${1} \
              --search-limit ${2} \
              --top-k 10 \
              --metric ip \
              --skip-build 1 \
              --index-file cpu_flat_${1}l_${2}q.faiss \
              --calc-recall true
}

run_ivf() {
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
            ./run_cpu \
              --index-type ivf \
              --dataset-dir /workspace/dataset/t2i \
              --learn-limit ${1} \
              --search-limit ${2} \
              --top-k 10 \
              --n-probe ${3} \
              --metric ip \
              --skip-build 1 \
              --index-file cpu_ivf_${1}l_${2}q.faiss \
              --calc-recall true
}

run_hnsw() {
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
            ./run_cpu \
              --index-type hnsw \
              --dataset-dir /workspace/dataset/t2i \
              --learn-limit ${1} \
              --search-limit ${2} \
              --top-k 10 \
              --ef ${3} \
              --metric ip \
              --skip-build 1 \
              --index-file cpu_hnsw_${1}l_${2}q.faiss \
              --calc-recall true
}

run_flat 100000   10000
run_flat 1000000  10000
run_flat 10000000 10000

run_ivf  100000   10000 256
run_ivf  1000000  10000 256
run_ivf  10000000 10000 256

run_hnsw 100000   10000 256
run_hnsw 1000000  10000 256
run_hnsw 10000000 10000 256
