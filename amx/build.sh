#!/bin/bash
set -e

g++ -std=c++23 -O3 run_amx.cc -ldnnl -fopenmp -march=sapphirerapids -mamx-bf16 -o run_amx
