#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

perf stat -e fp_arith_inst_retired.512b_packed_single ./run_faiss_cpu \
    --dataset-dir /workspace/dataset/gist \
    --learn-limit 1000000 \
    --search-limit 10000 \
    --top-k 10 \
    --ef 256
