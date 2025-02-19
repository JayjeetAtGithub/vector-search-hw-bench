#!/bin/bash
set -e

g++ -std=c++17 -O3 run_faiss_cpu.cc -ldnnl -o run_amx
