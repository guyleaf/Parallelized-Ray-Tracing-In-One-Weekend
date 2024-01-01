#!/usr/bin/env bash

export CMAKE_BUILD_TYPE="Release"
CPUS="0-9,20-29"
# CPUS="10-19,30-39"

SETUP="CXXFLAGS=-DMAX_DEPTH={max_depth} cmake -B build --fresh . && cmake --build build"
CMD="taskset -c ${CPUS} ./build/serial/InOneWeekend"
hyperfine -w 3 -r 5 -P max_depth 10 100 -D 10 --export-json=serial_max_depth.json --sort=command --shell=bash --setup "${SETUP}" "${CMD}"

SETUP="CXXFLAGS=-DSAMPLES_PER_PIXEL={samples_per_pixel} cmake -B build --fresh . && cmake --build build"
CMD="taskset -c ${CPUS} ./build/serial/InOneWeekend"
hyperfine -w 3 -r 5 -P samples_per_pixel 10 100 -D 10 --export-json=serial_samples_per_pixel.json --sort=command --shell=bash --setup "${SETUP}" "${CMD}"
