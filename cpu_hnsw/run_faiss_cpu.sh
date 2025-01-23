#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

perf stat -e fp_arith_inst_retired.512b_packed_single \
          -e fp_arith_inst_retired.vector \
          -e fp_arith_inst_retired.scalar_single \
          ./run_faiss_cpu \
            --dataset-dir /workspace/dataset/gist \
            --learn-limit 1000000 \
            --search-limit 10000 \
            --top-k 10 \
            --ef 256 \
            --metric 1
