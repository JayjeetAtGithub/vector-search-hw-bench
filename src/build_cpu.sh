#!/bin/bash
set -ex

g++ -std=c++23 -Ofast run_cpu.cc -lfaiss_avx512 -mavx512f -o run_cpu
