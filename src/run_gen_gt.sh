#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

run_gen_gt() {
    ./run_gen_gt \
        --dataset-dir /workspace/dataset/t2i \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k 10
}

run_gen_gt 100000 1000
run_gen_gt 100000 10000
run_gen_gt 100000 100000

run_gen_gt 1000000 1000
run_gen_gt 1000000 10000
run_gen_gt 1000000 100000

run_gen_gt 10000000 1000
run_gen_gt 10000000 10000
run_gen_gt 10000000 100000
