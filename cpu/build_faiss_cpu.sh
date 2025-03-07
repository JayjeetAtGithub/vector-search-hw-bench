#!/bin/bash
set -e

# we build faiss specifically for the Sapphire Rapids CPU with the avx512 and spr flags
g++ -std=c++17 -O3 run_faiss_cpu.cc -lfaiss_avx512_spr -o run_faiss_cpu
