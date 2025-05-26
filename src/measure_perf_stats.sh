#!/bin/bash
set -ex

echo "use numactcl to bind to CPU cores and memory node 0"

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

perf_stat() {
    perf stat -a \
        -e uncore_imc/cas_count_read/ \
        -e uncore_imc/cas_count_write/ \
        -e fp_arith_inst_retired.scalar_double \
        -e fp_arith_inst_retired.scalar_single \
        -e fp_arith_inst_retired.128b_packed_double \
        -e fp_arith_inst_retired.128b_packed_single \
        -e fp_arith_inst_retired.256b_packed_double \
        -e fp_arith_inst_retired.256b_packed_single \
        -e fp_arith_inst_retired.512b_packed_double \
        -e fp_arith_inst_retired.512b_packed_single \
        -e fp_arith_inst_retired.vector \
        -e cpu_clk_unhalted.thread \
        -e inst_retired.any \
        ${1}
}

perf_stat "./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/deep --learn-limit 128000000 --search-limit 10000 --top-k 10 --ef 650 --metric l2 --skip-build 1 --index-file cpu_hnsw_deep_128000000l_32m.faiss --calc-recall false"
perf_stat "./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/deep --learn-limit 64000000  --search-limit 10000 --top-k 10 --ef 550 --metric l2 --skip-build 1 --index-file cpu_hnsw_deep_64000000l_32m.faiss  --calc-recall false"
perf_stat "./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/deep --learn-limit 32000000  --search-limit 10000 --top-k 10 --ef 400 --metric l2 --skip-build 1 --index-file cpu_hnsw_deep_32000000l_32m.faiss  --calc-recall false"
perf_stat "./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/deep --learn-limit 16000000  --search-limit 10000 --top-k 10 --ef 320 --metric l2 --skip-build 1 --index-file cpu_hnsw_deep_16000000l_32m.faiss  --calc-recall false"
perf_stat "./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/deep --learn-limit 8000000   --search-limit 10000 --top-k 10 --ef 250 --metric l2 --skip-build 1 --index-file cpu_hnsw_deep_8000000l_32m.faiss   --calc-recall false"
perf_stat "./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/deep --learn-limit 4000000   --search-limit 10000 --top-k 10 --ef 200 --metric l2 --skip-build 1 --index-file cpu_hnsw_deep_4000000l_32m.faiss   --calc-recall false"
perf_stat "./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/deep --learn-limit 2000000   --search-limit 10000 --top-k 10 --ef 115 --metric l2 --skip-build 1 --index-file cpu_hnsw_deep_2000000l_32m.faiss   --calc-recall false"
perf_stat "./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/deep --learn-limit 1000000   --search-limit 10000 --top-k 10 --ef 110 --metric l2 --skip-build 1 --index-file cpu_hnsw_deep_1000000l_32m.faiss   --calc-recall false"
