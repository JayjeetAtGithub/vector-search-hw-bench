#!/bin/bash
set -e

g++ -std=c++17 -O3 run_faiss_gpu.cc -lfaiss_gpu_objs -o run_faiss_gpu
