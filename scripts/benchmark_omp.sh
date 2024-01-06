#!/usr/bin/env bash

export CMAKE_BUILD_TYPE="Release"
# CPUS="0-9,20-29"
# CPUS="10-19,30-39"

# CMD="OMP_NUM_THREADS={threads} taskset -c ${CPUS} ./build/openmp/OMPInOneWeekend"
CMD="OMP_NUM_THREADS={threads} ./build/openmp/OMPInOneWeekend"
SETUP='cmake -B build --fresh . && cmake --build build'
# SETUP='rm -r build && CXXFLAGS="-DUSE_FLOAT" cmake -B build --fresh . && cmake --build build'
hyperfine -w 3 -r 5 -P threads 1 32 -D 1 --export-json=omp_outer_dynamic_cyclic.json --sort=command --shell=bash --setup "${SETUP}" "${CMD}"
