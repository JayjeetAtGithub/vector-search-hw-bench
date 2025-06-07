#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

run_gen_gt() {
    ./run_gen_gt \
        --dataset-dir /workspace/dataset/deep \
        --learn-limit ${1} \
        --search-limit 10000 \
        --top-k 10 \
        --mem-type managed \
        --gt-file gt_${1}l_10000s_10k.bin
}

dataset_size=(1000000 2000000 4000000 8000000 16000000 32000000 64000000)

for size in "${dataset_size[@]}"; do
    run_gen_gt ${size}
done
