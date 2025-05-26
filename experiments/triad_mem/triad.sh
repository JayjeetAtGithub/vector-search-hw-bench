#!/bin/bash
set -e

g++ -std=c++23 -Ofast -fopenmp -funroll-all-loops triad.cc -o triad

# loop over number of cores
for i in $(seq 1 24); do
    echo "$i cores"
    numactl -C 0-$((i-1)) --membind 0 ./triad
done
