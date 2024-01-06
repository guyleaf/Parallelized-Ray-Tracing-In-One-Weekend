#!/usr/bin/env bash

export CMAKE_BUILD_TYPE="Release"
# CPUS="0-9,20-29"
# CPUS="10-19,30-39"

# CMD="taskset -c ${CPUS} ./build/cuda/CUDAInOneWeekend"
CMD="./build/cuda/CUDAInOneWeekend"
SETUP='CUDAFLAGS="-DCUDA_BLOCK_SIZE={block_size}" cmake -B build --fresh . && cmake --build build'
hyperfine -w 3 -r 5 -P block_size 1 32 -D 1 --export-json=cuda_block_size.json --sort=command --shell=bash --setup "${SETUP}" "${CMD}"
# hyperfine -w 3 -r 5 --shell=bash "taskset -c 10-19,30-39 ./build/cuda/CUDAInOneWeekend"
