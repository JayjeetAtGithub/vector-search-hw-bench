#!/bin/bash
set -e

g++ -std=c++17 -O3 run_cpu.cc -lfaiss_avx512 -o run_cpu
