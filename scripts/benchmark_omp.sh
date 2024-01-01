#!/usr/bin/env bash

CPUS="0-9,20-29"
# CPUS="10-19,30-39"
hyperfine -w 3 -r 5 -P threads 1 32 -D 1 --export-json=omp.json --sort=command --shell=bash "OMP_NUM_THREADS={threads} taskset -c ${CPUS} ./build/openmp/OMPInOneWeekend"
