#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

learn_limit=100000
search_limit=1000
calc_recall=${1:-false}

./run_faiss_gpu \
    --index-type ivf \
    --dataset-dir /workspace/dataset/deep1b \
    --learn-limit ${learn_limit} \
    --search-limit ${search_limit} \
    --top-k 10 \
    --n-probe 128 \
    --metric ip \
    --index-file gpu_ivf.faiss

sudo nsys profile \
    -t nvtx,cuda,osrt \
    -f true \
    --stats=true \
    --cuda-memory-usage=true \
    --cuda-um-cpu-page-faults=true \
    --cuda-event-trace=false \
    --cuda-um-gpu-page-faults=true \
    --gpu-metrics-devices 0 \
    --env-var CUDA_VISIBLE_DEVICES=0 \
    --output gpu_ivf_faiss \    
    ./run_faiss_gpu \
        --index-type ivf \
        --dataset-dir /workspace/dataset/deep1b \
        --learn-limit ${learn_limit} \
        --search-limit ${search_limit} \
        --top-k 10 \
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
    --metric ip \
    --index-file gpu_flat.faiss

sudo nsys profile \
    -t nvtx,cuda,osrt \
    -f true \
    --stats=true \
    --cuda-memory-usage=true \
    --cuda-um-cpu-page-faults=true \
    --cuda-event-trace=false \
    --cuda-um-gpu-page-faults=true \
    --gpu-metrics-devices 0 \
    --env-var CUDA_VISIBLE_DEVICES=0 \
    --output gpu_flat_faiss \    
    ./run_faiss_gpu \
        --index-type flat \
        --dataset-dir /workspace/dataset/deep1b \
        --learn-limit ${learn_limit} \
        --search-limit ${search_limit} \
        --top-k 10 \
        --metric ip \
        --skip-build 1 \
        --index-file gpu_flat.faiss \
        --calc-recall ${calc_recall}
