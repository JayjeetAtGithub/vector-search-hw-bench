#!/bin/bash
set -e

g++ -std=c++17 -O3 run_gpu.cc -lfaiss_avx512 -o run_gpu
g++ -std=c++17 -O3 run_gen_gt.cc  -lfaiss_avx512 -o run_gen_gt
