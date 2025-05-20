#!/bin/bash
set -e

# loop over number of cores
for i in $(seq 1 24); do
    echo "$i cores"
    numactl -C 0-$((i-1)) --membind 0 ./triad
done
