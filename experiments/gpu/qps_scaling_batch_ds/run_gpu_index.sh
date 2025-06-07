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
        --top-k 10 \
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
        --top-k 10 \
        --n-probe ${3} \
        --metric l2 \
        --skip-build 1 \
        --index-file ${INDEX_DIR}/gpu_ivf_deep_${1}l.faiss \
        --calc-recall true \
        --gt-file ${INDEX_DIR}/gt_${1}l_10000s_10k.bin
}


dataset_size=(1000000 2000000 4000000 8000000 16000000 32000000 64000000)
batch_size=(1 10 100 1000 10000)
for size in "${dataset_size[@]}"; do
    for batch in "${batch_size[@]}"; do
        run_flat ${size} ${batch}
    done
done

# Batch size = 1, Recall@10 = 95%
run_ivf 1000000   1 100
run_ivf 2000000   1 110
run_ivf 4000000   1 120
run_ivf 8000000   1 125
run_ivf 16000000  1 140
run_ivf 32000000  1 145
run_ivf 64000000  1 160

# Batch size = 10, Recall@10 = 95%
run_ivf 1000000   10 100
run_ivf 2000000   10 110
run_ivf 4000000   10 120
run_ivf 8000000   10 125
run_ivf 16000000  10 140
run_ivf 32000000  10 145
run_ivf 64000000  10 160

# Batch size = 100, Recall@10 = 95%
run_ivf 1000000   100  100
run_ivf 2000000   100  110
run_ivf 4000000   100  120
run_ivf 8000000   100  125
run_ivf 16000000  100  140
run_ivf 32000000  100  145
run_ivf 64000000  100  160

# Batch size = 1000, Recall@10 = 95%

run_ivf 1000000   1000  100
run_ivf 2000000   1000  110
run_ivf 4000000   1000  120
run_ivf 8000000   1000  125
run_ivf 16000000  1000  140
run_ivf 32000000  1000  145
run_ivf 64000000  1000  160

# Batch size = 10000, Recall@10 = 95%
run_ivf 1000000   10000  100
run_ivf 2000000   10000  110
run_ivf 4000000   10000  120
run_ivf 8000000   10000  125
run_ivf 16000000  10000  140
run_ivf 32000000  10000  145
run_ivf 64000000  10000  160
