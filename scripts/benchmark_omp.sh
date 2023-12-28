#!/usr/bin/env bash

hyperfine -w 3 -r 5 -P threads 1 10 -D 2 --sort=command --shell=bash "OMP_NUM_THREADS={threads} taskset -c 0-9,20-29 ./build/openmp/OMPInOneWeekend"
