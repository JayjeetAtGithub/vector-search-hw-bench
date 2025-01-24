#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

learn_limit=100000
search_limit=1000

./run_faiss_cpu \
    --index-type hnsw \
    --dataset-dir /workspace/dataset/deep1b \
    --learn-limit ${learn_limit} \
    --search-limit ${search_limit} \
    --top-k 10 \
    --ef 256 \
    --metric ip \
    --index-file cpu_hnsw.faiss

perf stat -e fp_arith_inst_retired.512b_packed_single \
          -e fp_arith_inst_retired.vector \
          -e fp_arith_inst_retired.scalar_single \
          ./run_faiss_cpu \
            --index-type hnsw \
            --dataset-dir /workspace/dataset/deep1b \
            --learn-limit ${learn_limit} \
            --search-limit ${search_limit} \
            --top-k 10 \
            --ef 256 \
            --metric ip \
            --skip-build 1 \
            --index-file cpu_hnsw.faiss \
            --calc-recall

./run_faiss_cpu \
    --index-type ivf \
    --dataset-dir /workspace/dataset/deep1b \
    --learn-limit ${learn_limit} \
    --search-limit ${search_limit} \
    --top-k 10 \
    --ef 256 \
    --metric ip \
    --index-file cpu_ivf.faiss

perf stat -e fp_arith_inst_retired.512b_packed_single \
          -e fp_arith_inst_retired.vector \
          -e fp_arith_inst_retired.scalar_single \
          ./run_faiss_cpu \
            --index-type ivf \
            --dataset-dir /workspace/dataset/deep1b \
            --learn-limit ${learn_limit} \
            --search-limit ${search_limit} \
            --top-k 10 \
            --ef 256 \
            --metric ip \
            --skip-build 1 \
            --index-file cpu_ivf.faiss \
            --calc-recall

./run_faiss_cpu \
    --index-type flat \
    --dataset-dir /workspace/dataset/deep1b \
    --learn-limit ${learn_limit} \
    --search-limit ${search_limit} \
    --top-k 10 \
    --ef 256 \
    --metric ip \
    --index-file cpu_flat.faiss

perf stat -e fp_arith_inst_retired.512b_packed_single \
          -e fp_arith_inst_retired.vector \
          -e fp_arith_inst_retired.scalar_single \
          ./run_faiss_cpu \
            --index-type flat \
            --dataset-dir /workspace/dataset/deep1b \
            --learn-limit ${learn_limit} \
            --search-limit ${search_limit} \
            --top-k 10 \
            --ef 256 \
            --metric ip \
            --skip-build 1 \
            --index-file cpu_flat.faiss \
            --calc-recall
