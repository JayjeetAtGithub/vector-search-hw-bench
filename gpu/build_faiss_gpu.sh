#!/bin/bash
set -e

g++ -std=c++17 -O3 run_faiss_gpu.cc -lfaiss -o run_faiss_gpu
