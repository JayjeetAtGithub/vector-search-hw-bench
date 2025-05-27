#!/bin/bash
set -ex

numactl -C 0-23 --membind 0 ./perf_stats.sh
