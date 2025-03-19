#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

run_flat() {
    ./run_amx \
        --index-type flat \
        --dataset-dir /workspace/dataset/t2i \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k 10 \
        --calc-recall true
}

run_flat 100000 10
run_flat 100000 100
run_flat 100000 1000
run_flat 100000 10000

run_flat 1000000 10
run_flat 1000000 100
run_flat 1000000 1000
run_flat 1000000 10000

run_flat 10000000 10
run_flat 10000000 100
run_flat 10000000 1000
run_flat 10000000 10000
