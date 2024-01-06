#!/usr/bin/env bash
set -eu

export CMAKE_BUILD_TYPE="Release"
# CPUS="0-9,20-29"
# CPUS="10-19,30-39"

SETUP='cmake -B build --fresh . && cmake --build build'
# CMD="taskset -c ${CPUS} ./build/serial/InOneWeekend"
CMD="./build/serial/InOneWeekend"
hyperfine -w 3 -r 5 --export-json=serial.json --sort=command --shell=bash --setup "${SETUP}" "${CMD}"

# SETUP='CXXFLAGS="-DMAP_SIZE={map_size}" cmake -B build --fresh . && cmake --build build'
# hyperfine -w 3 -r 5 -P map_size 2 32 -D 2 --export-json=serial_map_size.json --sort=command --shell=bash --setup "${SETUP}" "${CMD}"

# SETUP="CXXFLAGS=-DMAX_DEPTH={max_depth} cmake -B build --fresh . && cmake --build build"
# CMD="taskset -c ${CPUS} ./build/serial/InOneWeekend"
# hyperfine -w 3 -r 5 -P max_depth 10 100 -D 10 --export-json=serial_max_depth.json --sort=command --shell=bash --setup "${SETUP}" "${CMD}"

# SETUP="CXXFLAGS=-DIMAGE_WIDTH={image_width} cmake -B build --fresh . && cmake --build build"
# CMD="taskset -c ${CPUS} ./build/serial/InOneWeekend"
# hyperfine -w 3 -r 5 -L image_width 1200,1280,1920,2560,3840 --export-json=serial_image_width.json --sort=command --shell=bash --setup "${SETUP}" "${CMD}"

# SETUP="CXXFLAGS=-DSAMPLES_PER_PIXEL={samples_per_pixel} cmake -B build --fresh . && cmake --build build"
# CMD="taskset -c ${CPUS} ./build/serial/InOneWeekend"
# hyperfine -w 3 -r 5 -P samples_per_pixel 10 100 -D 10 --export-json=serial_samples_per_pixel.json --sort=command --shell=bash --setup "${SETUP}" "${CMD}"
