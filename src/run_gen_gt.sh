#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

run_gen_gt() {
    ./run_gen_gt \
        --dataset-dir /workspace/dataset/t2i \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k ${3} \
        --gt-file gt_${1}l_${2}s_${3}k.bin
}

run_gen_gt 1000000  10000 10
run_gen_gt 2000000  10000 10
run_gen_gt 4000000  10000 10
run_gen_gt 8000000  10000 10
run_gen_gt 16000000 10000 10
run_gen_gt 32000000 10000 10

run_gen_gt 1000000  10000 100
run_gen_gt 2000000  10000 100
run_gen_gt 4000000  10000 100
run_gen_gt 8000000  10000 100
run_gen_gt 16000000 10000 100
run_gen_gt 32000000 10000 100
