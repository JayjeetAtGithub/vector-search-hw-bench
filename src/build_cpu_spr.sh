#!/bin/bash
set -ex

g++ -std=c++20 -Ofast -march=sapphirerapids run_cpu.cc -lfaiss_avx512_spr -o run_cpu
