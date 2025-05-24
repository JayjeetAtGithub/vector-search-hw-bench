#!/bin/bash
set -e

perf_stat() {
    perf stat -a -e uncore_imc/cas_count_read/ -e uncore_imc/cas_count_write/ "${1}"
}

perf_stat ./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/t2i --learn-limit 32000000 --search-limit 100000 --top-k 10 --ef 2250 --metric ip --skip-build 1 --index-file cpu_hnsw_32000000l.faiss --calc-recall true --gt-file gt_32000000l_100000s_10k.bin
perf_stat ./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/t2i --learn-limit 16000000 --search-limit 100000 --top-k 10 --ef 1650 --metric ip --skip-build 1 --index-file cpu_hnsw_16000000l.faiss --calc-recall true --gt-file gt_16000000l_100000s_10k.bin
perf_stat ./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/t2i --learn-limit 8000000 --search-limit 100000 --top-k 10 --ef 1250 --metric ip --skip-build 1 --index-file cpu_hnsw_8000000l.faiss --calc-recall true --gt-file gt_8000000l_100000s_10k.bin
perf_stat ./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/t2i --learn-limit 4000000 --search-limit 100000 --top-k 10 --ef 850 --metric ip --skip-build 1 --index-file cpu_hnsw_4000000l.faiss --calc-recall true --gt-file gt_4000000l_100000s_10k.bin
perf_stat ./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/t2i --learn-limit 2000000 --search-limit 100000 --top-k 10 --ef 450 --metric ip --skip-build 1 --index-file cpu_hnsw_2000000l.faiss --calc-recall true --gt-file gt_2000000l_100000s_10k.bin
perf_stat ./run_cpu --index-type hnsw --dataset-dir /workspace/dataset/t2i --learn-limit 1000000 --search-limit 100000 --top-k 10 --ef 250 --metric ip --skip-build 1 --index-file cpu_hnsw_1000000l.faiss --calc-recall true --gt-file gt_1000000l_100000s_10k.bin