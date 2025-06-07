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
        --search-limit ${2} \
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
        --search-limit ${2} \
        --top-k ${3} \
        --n-probe ${4} \
        --metric l2 \
        --skip-build 1 \
        --index-file ${INDEX_DIR}/gpu_ivf_deep_${1}l.faiss \
        --calc-recall true \
        --gt-file ${INDEX_DIR}/gt_${1}l_${2}s_${3}k.bin
}

# Batch size = 100, Recall@10 = 95%
# run_flat 1000000  100 10
# run_flat 2000000  100 10
# run_flat 4000000  100 10
# run_flat 8000000  100 10
# run_flat 16000000 100 10
# run_flat 32000000 100 10
run_ivf 1000000   100 10 75
run_ivf 2000000   100 10 78
run_ivf 4000000   100 10 80
run_ivf 8000000   100 10 80
run_ivf 16000000  100 10 80
run_ivf 32000000  100 10 95
run_ivf 64000000  100 10 100

# Batch size = 1000, Recall@10 = 95%
# run_flat 1000000  1000 10
# run_flat 2000000  1000 10
# run_flat 4000000  1000 10
# run_flat 8000000  1000 10
# run_flat 16000000 1000 10
# run_flat 32000000 1000 10
run_ivf 1000000   1000 10 80
run_ivf 2000000   1000 10 90
run_ivf 4000000   1000 10 100
run_ivf 8000000   1000 10 105
run_ivf 16000000  1000 10 110
run_ivf 32000000  1000 10 115
run_ivf 64000000  1000 10 120

# Batch size = 10000, Recall@10 = 95%
# run_flat 1000000  10000 10
# run_flat 2000000  10000 10
# run_flat 4000000  10000 10
# run_flat 8000000  10000 10
# run_flat 16000000 10000 10
# run_flat 32000000 10000 10
run_ivf 1000000   10000 10 80
run_ivf 2000000   10000 10 90
run_ivf 4000000   10000 10 100
run_ivf 8000000   10000 10 110
run_ivf 16000000  10000 10 115
run_ivf 32000000  10000 10 120
run_ivf 64000000  10000 10 125

# Batch size = 100000, Recall@10 = 95%
# run_flat 1000000  100000 10
# run_flat 2000000  100000 10
# run_flat 4000000  100000 10
# run_flat 8000000  100000 10
# run_flat 16000000 100000 10
# run_flat 32000000 100000 10
run_ivf 1000000   100000 10 80
run_ivf 2000000   100000 10 90
run_ivf 4000000   100000 10 100
run_ivf 8000000   100000 10 105
run_ivf 16000000  100000 10 110
run_ivf 32000000  100000 10 115
run_ivf 64000000  100000 10 120
