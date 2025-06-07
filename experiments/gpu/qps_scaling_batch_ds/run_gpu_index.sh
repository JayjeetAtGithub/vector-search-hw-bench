#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

BINARY_DIR="/workspace/vector-search-hw-bench/src/"
INDEX_DIR="/workspace/vector-search-hw-bench/src/"

run_flat() {
    ${BINARY_DIR}/run_gpu \
        --index-type flat \
        --dataset-dir /workspace/dataset/deep \
        --learn-limit ${1} \
        --search-limit 10000 \
        --batch-size ${2} \
        --top-k ${3} \
        --metric l2 \
        --skip-build 1 \
        --index-file ${INDEX_DIR}/gpu_flat_deep_${1}l.faiss \
        --calc-recall false
}

run_ivf() {
    ${BINARY_DIR}/run_gpu \
        --index-type ivf \
        --dataset-dir /workspace/dataset/deep \
        --learn-limit ${1} \
        --search-limit 10000 \
        --batch-size ${2} \
        --top-k ${3} \
        --n-probe ${4} \
        --metric l2 \
        --skip-build 1 \
        --index-file ${INDEX_DIR}/gpu_ivf_deep_${1}l.faiss \
        --calc-recall true \
        --gt-file ${INDEX_DIR}/gt_${1}l_10000s_${3}k.bin
}

run_ivf 64000000  10 10 160
run_ivf 64000000  100 10 160
run_ivf 64000000  500 10 160
run_ivf 64000000  1000 10 160
run_ivf 64000000  2000 10 160
run_ivf 64000000  5000 10 160
run_ivf 64000000  10000 10 160

# # Batch size = 100, Recall@10 = 95%
# run_flat 1000000  100 10
# run_flat 2000000  100 10
# run_flat 4000000  100 10
# run_flat 8000000  100 10
# run_flat 16000000 100 10
# run_flat 32000000 100 10
# run_ivf 1000000   100 10 100
# run_ivf 2000000   100 10 110
# run_ivf 4000000   100 10 120
# run_ivf 8000000   100 10 125
# run_ivf 16000000  100 10 140
# run_ivf 32000000  100 10 145
# run_ivf 64000000  100 10 160

# # Batch size = 500, Recall@10 = 95%
# run_flat 1000000  500 10
# run_flat 2000000  500 10
# run_flat 4000000  500 10
# run_flat 8000000  500 10
# run_flat 16000000 500 10
# run_flat 32000000 500 10
# run_ivf 1000000   500 10 100
# run_ivf 2000000   500 10 110
# run_ivf 4000000   500 10 120
# run_ivf 8000000   500 10 125
# run_ivf 16000000  500 10 140
# run_ivf 32000000  500 10 145
# run_ivf 64000000  500 10 160

# # Batch size = 1000, Recall@10 = 95%
# run_flat 1000000  1000 10
# run_flat 2000000  1000 10
# run_flat 4000000  1000 10
# run_flat 8000000  1000 10
# run_flat 16000000 1000 10
# run_flat 32000000 1000 10
# run_ivf 1000000   1000 10 100
# run_ivf 2000000   1000 10 110
# run_ivf 4000000   1000 10 120
# run_ivf 8000000   1000 10 125
# run_ivf 16000000  1000 10 140
# run_ivf 32000000  1000 10 145
# run_ivf 64000000  1000 10 160

# # Batch size = 2000, Recall@10 = 95%
# run_flat 1000000  2000 10
# run_flat 2000000  2000 10
# run_flat 4000000  2000 10
# run_flat 8000000  2000 10
# run_flat 16000000 2000 10
# run_flat 32000000 2000 10
# run_ivf 1000000   2000 10 100
# run_ivf 2000000   2000 10 110
# run_ivf 4000000   2000 10 120
# run_ivf 8000000   2000 10 125
# run_ivf 16000000  2000 10 140
# run_ivf 32000000  2000 10 145
# run_ivf 64000000  2000 10 160

# # Batch size = 5000, Recall@10 = 95%
# run_flat 1000000  5000 10
# run_flat 2000000  5000 10
# run_flat 4000000  5000 10
# run_flat 8000000  5000 10
# run_flat 16000000 5000 10
# run_flat 32000000 5000 10
# run_ivf 1000000   5000 10 100
# run_ivf 2000000   5000 10 110
# run_ivf 4000000   5000 10 120
# run_ivf 8000000   5000 10 125
# run_ivf 16000000  5000 10 140
# run_ivf 32000000  5000 10 145
# run_ivf 64000000  5000 10 160

# # Batch size = 10000, Recall@10 = 95%
# run_flat 1000000  10000 10
# run_flat 2000000  10000 10
# run_flat 4000000  10000 10
# run_flat 8000000  10000 10
# run_flat 16000000 10000 10
# run_flat 32000000 10000 10
# run_ivf 1000000   10000 10 100
# run_ivf 2000000   10000 10 110
# run_ivf 4000000   10000 10 120
# run_ivf 8000000   10000 10 125
# run_ivf 16000000  10000 10 140
# run_ivf 32000000  10000 10 145
# run_ivf 64000000  10000 10 160
