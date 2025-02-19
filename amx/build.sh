#!/bin/bash
set -e

g++ -std=c++23 -O3 run_amx.cc -ldnnl -o run_amx
