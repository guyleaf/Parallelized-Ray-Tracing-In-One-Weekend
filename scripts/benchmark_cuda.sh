#!/usr/bin/env bash

hyperfine -w 3 -r 5 --shell=bash "taskset -c 0-9,20-29 ./build/cuda/CUDAInOneWeekend"
# hyperfine -w 3 -r 5 --shell=bash "taskset -c 10-19,30-39 ./build/cuda/CUDAInOneWeekend"
