#!/usr/bin/env bash

hyperfine -w 3 -r 5 -P threads 1 16 -D 2 --sort=command --shell=bash "OMP_NUM_THREADS={threads} taskset -c 0-9,20-29 ./build/openmp/OMPInOneWeekend"
# hyperfine -w 3 -r 5 -P threads 1 16 -D 2 --sort=command --shell=bash "OMP_NUM_THREADS={threads} taskset -c 10-19,30-39 ./build/openmp/OMPInOneWeekend"
