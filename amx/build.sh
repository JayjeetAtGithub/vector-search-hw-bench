#!/bin/bash
set -e

g++ -std=c++17 -O3 run_amx.cc -ldnnl -o run_amx
