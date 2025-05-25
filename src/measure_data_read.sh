#!/bin/bash
set -ex

perf_stat() {
    perf stat -a -e uncore_imc/cas_count_read/ -e uncore_imc/cas_count_write/ ${1}
}

perf_stat "./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/deep --learn-limit 128000000 --search-limit 10000 --top-k 10 --ef 650 --metric l2 --skip-build 1 --index-file cpu_hnsw_128000000l.faiss --calc-recall true"
perf_stat "./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/deep --learn-limit 64000000  --search-limit 10000 --top-k 10 --ef 550 --metric l2 --skip-build 1 --index-file cpu_hnsw_64000000l.faiss  --calc-recall true"
perf_stat "./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/deep --learn-limit 32000000  --search-limit 10000 --top-k 10 --ef 400 --metric l2 --skip-build 1 --index-file cpu_hnsw_32000000l.faiss  --calc-recall true"
perf_stat "./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/deep --learn-limit 16000000  --search-limit 10000 --top-k 10 --ef 320 --metric l2 --skip-build 1 --index-file cpu_hnsw_16000000l.faiss  --calc-recall true"
perf_stat "./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/deep --learn-limit 8000000   --search-limit 10000 --top-k 10 --ef 250 --metric l2 --skip-build 1 --index-file cpu_hnsw_8000000l.faiss   --calc-recall true"
perf_stat "./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/deep --learn-limit 4000000   --search-limit 10000 --top-k 10 --ef 200 --metric l2 --skip-build 1 --index-file cpu_hnsw_4000000l.faiss   --calc-recall true"
perf_stat "./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/deep --learn-limit 2000000   --search-limit 10000 --top-k 10 --ef 115 --metric l2 --skip-build 1 --index-file cpu_hnsw_2000000l.faiss   --calc-recall true"
perf_stat "./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/deep --learn-limit 1000000   --search-limit 10000 --top-k 10 --ef 110 --metric l2 --skip-build 1 --index-file cpu_hnsw_1000000l.faiss   --calc-recall true"
