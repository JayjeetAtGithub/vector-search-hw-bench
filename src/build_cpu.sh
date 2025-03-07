#!/bin/bash
set -e

# we build faiss specifically for the Sapphire Rapids CPU with the avx512 and spr flags
g++ -std=c++17 -O3 run_cpu.cc -lfaiss_avx512_spr -march=sapphirerapids -o run_cpu
