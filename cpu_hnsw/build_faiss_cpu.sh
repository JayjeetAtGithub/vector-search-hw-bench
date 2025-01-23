#!/bin/bash
set -e

g++ -std=c++17 -O3 run_faiss_cpu.cc -lfaiss_avx512_spr -o run_faiss_cpu
