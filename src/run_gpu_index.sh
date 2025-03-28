#!/bin/bash
set -e

datetime=$(date +"%Y-%m-%d %H:%M:%S")
cpu=$(lscpu | sed -nr '/Model name/ s/.*:\s*(.*) @ .*/\1/p')
cores=$(lscpu | sed -nr '/^CPU\(s\)/ s/.*:\s*(.*)/\1/p')
arch=$(lscpu | sed -nr '/Architecture/ s/.*:\s*(.*)/\1/p')
threads_per_core=$(lscpu | sed -nr '/Thread\(s\) per core/ s/.*:\s*(.*)/\1/p')
cores_per_socket=$(lscpu | sed -nr '/Core\(s\) per socket/ s/.*:\s*(.*)/\1/p')
sockets=$(lscpu | sed -nr '/Socket\(s\)/ s/.*:\s*(.*)/\1/p')
l1d_cache=$(lscpu | sed -nr '/L1d cache/ s/.*:\s*(.*)/\1/p')
l1i_cache=$(lscpu | sed -nr '/L1i cache/ s/.*:\s*(.*)/\1/p')
l2_cache=$(lscpu | sed -nr '/L2 cache/ s/.*:\s*(.*)/\1/p')
l3_cache=$(lscpu | sed -nr '/L3 cache/ s/.*:\s*(.*)/\1/p')
dram=$(free -h | sed -nr '/Mem:/ s/.*:\s*(.*)/\1/p' | cut -d' ' -f1)

echo "------------------------------------------------------------------------"
echo "Running GPU index benchmark on Yandex T2I dataset at $datetime"
echo "------------------------------------------------------------------------"
echo " "
echo "Arch:  $arch"
echo "Model: $cpu"
echo "Cores: $cores"
echo "Sockets: $sockets"
echo "Threads per core: $threads_per_core"
echo "Cores per socket: $cores_per_socket"
echo "L1d cache: $l1d_cache"
echo "L1i cache: $l1i_cache"
echo "L2 cache: $l2_cache"
echo "L3 cache: $l3_cache"
echo "DRAM: $dram"
echo " "
echo "------------------------------------------------------------------------"
echo " "
nvidia-smi
echo " "
echo "------------------------------------------------------------------------"

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

run_flat() {
    ./run_gpu \
        --index-type flat \
        --dataset-dir /workspace/dataset/t2i \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k 10 \
        --metric ip \
        --skip-build 1 \
        --index-file gpu_flat_${1}l.faiss \
        --calc-recall true \
        --gt-file gt_${1}l_${2}s.bin
}

run_ivf() {
    ./run_gpu \
        --index-type ivf \
        --dataset-dir /workspace/dataset/t2i \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k 10 \
        --n-probe ${3} \
        --metric ip \
        --skip-build 1 \
        --index-file gpu_ivf_${1}l.faiss \
        --calc-recall true \
        --gt-file gt_${1}l_${2}s.bin
}

# IVF is at 91-92% recall with the given configuration

run_flat 100000 1000
run_flat 100000 10000
run_flat 100000 100000

run_ivf  100000 1000   64
run_ivf  100000 10000  64
run_ivf  100000 100000 64

run_flat 1000000 1000
run_flat 1000000 10000
run_flat 1000000 100000

run_ivf  1000000 1000   96
run_ivf  1000000 10000  96
run_ivf  1000000 100000 96

run_flat 10000000 1000
run_flat 10000000 10000
run_flat 10000000 100000

run_ivf  10000000 1000   128
run_ivf  10000000 10000  128
run_ivf  10000000 100000 128
