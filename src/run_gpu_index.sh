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
        --top-k ${3} \
        --metric ip \
        --skip-build 1 \
        --index-file gpu_flat_${1}l.faiss \
        --calc-recall true \
        --gt-file gt_${1}l_${2}s_${3}k.bin
}

run_ivf() {
    ./run_gpu \
        --index-type ivf \
        --dataset-dir /workspace/dataset/t2i \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k ${3} \
        --n-probe ${4} \
        --metric ip \
        --skip-build 1 \
        --index-file gpu_ivf_${1}l.faiss \
        --calc-recall true \
        --gt-file gt_${1}l_${2}s_${3}k.bin
}

# Batch size = 100, Recall@10 = 90-91%
# run_flat 1000000  100 10
# run_flat 2000000  100 10
# run_flat 4000000  100 10
# run_flat 8000000  100 10
# run_flat 16000000 100 10
# run_flat 32000000 100 10
run_ivf 1000000   100 10 72
run_ivf 2000000   100 10 78
run_ivf 4000000   100 10 80
run_ivf 8000000   100 10 82
run_ivf 16000000  100 10 85
run_ivf 32000000  100 10 95

# Batch size = 1000, Recall@10 = 90-91%
# run_flat 1000000  1000 10
# run_flat 2000000  1000 10
# run_flat 4000000  1000 10
# run_flat 8000000  1000 10
# run_flat 16000000 1000 10
# run_flat 32000000 1000 10
run_ivf 1000000   1000 10 80
run_ivf 2000000   1000 10 90
run_ivf 4000000   1000 10 100
run_ivf 8000000   1000 10 105
run_ivf 16000000  1000 10 110
run_ivf 32000000  1000 10 115

# Batch size = 10000, Recall@10 = 90-91%
# run_flat 1000000  10000 10
# run_flat 2000000  10000 10
# run_flat 4000000  10000 10
# run_flat 8000000  10000 10
# run_flat 16000000 10000 10
# run_flat 32000000 10000 10
run_ivf 1000000   10000 10 80
run_ivf 2000000   10000 10 90
run_ivf 4000000   10000 10 100
run_ivf 8000000   10000 10 110
run_ivf 16000000  10000 10 115
run_ivf 32000000  10000 10 120

# Batch size = 100000, Recall@10 = 90-91%
# run_flat 1000000  100000 10
# run_flat 2000000  100000 10
# run_flat 4000000  100000 10
# run_flat 8000000  100000 10
# run_flat 16000000 100000 10
# run_flat 32000000 100000 10
run_ivf 1000000   100000 10 80
run_ivf 2000000   100000 10 90
run_ivf 4000000   100000 10 100
run_ivf 8000000   100000 10 105
run_ivf 16000000  100000 10 110
run_ivf 32000000  100000 10 115
