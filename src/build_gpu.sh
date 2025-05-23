#!/bin/bash
set -ex

g++ -std=c++23 -Ofast -fno-omit-frame-pointer run_gpu.cc -lfaiss_avx512 -o run_gpu
g++ -std=c++23 -Ofast -fno-omit-frame-pointer run_gen_gt.cc  -lfaiss_avx512 -o run_gen_gt
