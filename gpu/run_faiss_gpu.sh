#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

learn_limit=1000000
search_limit=10000
calc_recall=${1:-false}

./run_faiss_gpu \
    --index-type ivf \
    --dataset-dir /workspace/dataset/deep1b \
    --learn-limit ${learn_limit} \
    --search-limit ${search_limit} \
    --top-k 10 \
    --ef 256 \
    --n-probe 128 \
    --metric ip \
    --index-file gpu_ivf.faiss

./run_faiss_gpu \
    --index-type ivf \
    --dataset-dir /workspace/dataset/deep1b \
    --learn-limit ${learn_limit} \
    --search-limit ${search_limit} \
    --top-k 10 \
    --ef 256 \
    --n-probe 128 \
    --metric ip \
    --skip-build 1 \
    --index-file gpu_ivf.faiss \
    --calc-recall ${calc_recall}

./run_faiss_gpu \
    --index-type flat \
    --dataset-dir /workspace/dataset/deep1b \
    --learn-limit ${learn_limit} \
    --search-limit ${search_limit} \
    --top-k 10 \
    --ef 256 \
    --metric ip \
    --index-file gpu_flat.faiss

./run_faiss_gpu \
    --index-type flat \
    --dataset-dir /workspace/dataset/deep1b \
    --learn-limit ${learn_limit} \
    --search-limit ${search_limit} \
    --top-k 10 \
    --ef 256 \
    --metric ip \
    --skip-build 1 \
    --index-file gpu_flat.faiss \
    --calc-recall ${calc_recall}
