#!/bin/bash
set -e

g++ -std=c++20 -Ofast run_cpu.cc -lfaiss_avx512 -o run_cpu
