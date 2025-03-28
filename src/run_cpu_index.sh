#!/bin/bash
set -e

datetime=$(date +"%Y-%m-%d %H:%M:%S")

echo "---------------------------------------------------"
echo "Running CPU index benchmark on Yandex T2I dataset at $datetime"
echo "---------------------------------------------------"

cpu=$(lscpu | sed -nr '/Model name/ s/.*:\s*(.*) @ .*/\1/p')
cores=$(lscpu | sed -nr '/^CPU\(s\)/ s/.*:\s*(.*)/\1/p')
arch=$(lscpu | sed -nr '/Architecture/ s/.*:\s*(.*)/\1/p')
threads_per_core=$(lscpu | sed -nr '/Thread\(s\) per core/ s/.*:\s*(.*)/\1/p')
cores_per_socket=$(lscpu | sed -nr '/Core\(s\) per socket/ s/.*:\s*(.*)/\1/p')
sockets=$(lscpu | sed -nr '/Socket\(s\)/ s/.*:\s*(.*)/\1/p')

echo "CPU Arch:  $arch"
echo "CPU Model: $cpu"
echo "CPU Cores: $cores"
echo "CPU Sockets: $sockets"
echo "CPU Threads per core: $threads_per_core"
echo "CPU Cores per socket: $cores_per_socket"
echo "---------------------------------------------------"

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

run_flat() {
    ./run_cpu \
        --index-type flat \
        --dataset-dir /workspace/dataset/t2i \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k 10 \
        --metric ip \
        --skip-build 1 \
        --index-file cpu_flat_${1}l.faiss \
        --calc-recall true \
        --gt-file gt_${1}l_${2}s.bin
}

run_ivf() {
    ./run_cpu \
        --index-type ivf \
        --dataset-dir /workspace/dataset/t2i \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k 10 \
        --n-probe ${3} \
        --metric ip \
        --skip-build 1 \
        --index-file cpu_ivf_${1}l.faiss \
        --calc-recall true \
        --gt-file gt_${1}l_${2}s.bin
}

run_hnsw() {
    ./run_cpu \
        --index-type hnsw \
        --dataset-dir /workspace/dataset/t2i \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k 10 \
        --ef ${3} \
        --metric ip \
        --skip-build 1 \
        --index-file cpu_hnsw_${1}l.faiss \
        --calc-recall true \
        --gt-file gt_${1}l_${2}s.bin
}

# IVF is at 91-92% recall with the given configuration
# HNSW is at 90% recall with the given configuration

# run_flat 100000 100
# run_flat 100000 1000
# run_flat 100000 10000

run_ivf  100000 1000   64
run_ivf  100000 10000  64
run_ivf  100000 100000 64
 
run_hnsw 100000 1000   64
run_hnsw 100000 10000  64
run_hnsw 100000 100000 64

# run_flat 1000000 100
# run_flat 1000000 1000
# run_flat 1000000 10000

run_ivf  1000000 1000   96
run_ivf  1000000 10000  96
run_ivf  1000000 100000 96

run_hnsw 1000000 1000   256
run_hnsw 1000000 10000  256
run_hnsw 1000000 100000 256

# run_flat 10000000 100
# run_flat 10000000 1000
# run_flat 10000000 10000

run_ivf  10000000 1000   128
run_ivf  10000000 10000  128
run_ivf  10000000 100000 128

run_hnsw 10000000 1000   1024
run_hnsw 10000000 10000  1024
run_hnsw 10000000 100000 1024
