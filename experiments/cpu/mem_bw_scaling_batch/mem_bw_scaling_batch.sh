#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

run_hnsw_deep() {
    pcm-memory 0.5 \
        -csv="pcm_bw_hnsw_deep_${2}s.pcm_log" \
        -- ./runner.sh ${1} ${2} ${3} ${4} ${5} 
}

run_hnsw_deep 128000000 100 10 650 32
run_hnsw_deep 128000000 500 10 650 32
run_hnsw_deep 128000000 1000 10 650 32
run_hnsw_deep 128000000 2000 10 650 32
run_hnsw_deep 128000000 5000 10 650 32
run_hnsw_deep 128000000 10000 10 650 32
