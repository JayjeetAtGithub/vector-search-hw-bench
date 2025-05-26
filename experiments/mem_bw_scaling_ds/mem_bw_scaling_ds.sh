#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

run_hnsw_deep() {
    pcm-memory 0.5 \
        -csv="pcm_bw_hnsw_deep_${1}l.pcm_log" \
        -- ./runner.sh ${1} ${2} ${3} ${4} ${5}
}

run_hnsw_deep 1000000   10000 10 110  32
run_hnsw_deep 2000000   10000 10 115  32
run_hnsw_deep 4000000   10000 10 200  32
run_hnsw_deep 8000000   10000 10 250 32
run_hnsw_deep 16000000  10000 10 320 32
run_hnsw_deep 32000000  10000 10 400 32
run_hnsw_deep 64000000  10000 10 550 32
run_hnsw_deep 128000000 10000 10 650 32
