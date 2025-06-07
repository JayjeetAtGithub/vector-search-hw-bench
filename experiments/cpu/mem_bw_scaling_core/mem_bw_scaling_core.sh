#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

run_hnsw_deep() {
    pcm-memory 0.5 \
        -csv="pcm_bw_hnsw_deep_${6}c.pcm_log" \
        -- ./runner.sh ${1} ${2} ${3} ${4} ${5} ${6}
}

run_hnsw_deep 128000000 10000 10 650 32 0-0
run_hnsw_deep 128000000 10000 10 650 32 0-1
run_hnsw_deep 128000000 10000 10 650 32 0-3
run_hnsw_deep 128000000 10000 10 650 32 0-7
run_hnsw_deep 128000000 10000 10 650 32 0-15
run_hnsw_deep 128000000 10000 10 650 32 0-23
