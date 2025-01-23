#!/bin/bash
set -e

./run_faiss_cpu \
    --dataset-dir /workspace/dataset/gist \
    --query-dir /workspace/dataset/gist \
    --learn-limit 1000000 \
    --search-limit 10000 \
    --top-k 10 \
    --ef 256

