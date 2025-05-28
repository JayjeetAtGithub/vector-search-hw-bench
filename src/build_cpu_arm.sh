#!/bin/bash
set -ex

g++ -std=c++23 -Ofast run_cpu.cc -lfaiss_sve -march=armv8-a+sve2 -o run_cpu
