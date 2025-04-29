#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

run_gen_gt() {
    ./run_gen_gt \
        --dataset-dir /workspace/dataset/t2i \
        --learn-limit ${1} \
        --search-limit ${2} \
        --top-k ${3} \
        --gt-file gt_${1}l_${2}s_${3}k.bin
}

dataset_size=(1000000 2000000 4000000 8000000 16000000 32000000)
batch_size=(100 1000 10000 100000)
top_k=(10 100)

for size in "${dataset_size[@]}"; do
    for batch in "${batch_size[@]}"; do
        for k in "${top_k[@]}"; do
            run_gen_gt ${size} ${batch} ${k}
        done
    done
done
