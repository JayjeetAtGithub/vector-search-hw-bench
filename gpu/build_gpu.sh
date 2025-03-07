#!/bin/bash
set -e

g++ -std=c++17 -O3 run_gpu.cc -lfaiss -o run_gpu
