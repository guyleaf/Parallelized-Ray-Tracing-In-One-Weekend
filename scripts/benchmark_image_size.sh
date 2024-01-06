#!/usr/bin/env bash

export CMAKE_BUILD_TYPE="Release"
# CPUS="0-9,20-29"
CPUS="10-19,30-39"

CMD1="taskset -c ${CPUS} ./build/serial/InOneWeekend"
CMD2="OMP_NUM_THREADS=22 taskset -c ${CPUS} ./build/openmp/OMPInOneWeekend"
CMD3="taskset -c ${CPUS} ./build/cuda/CUDAInOneWeekend"
SETUP='CUDAFLAGS="-DCUDA_BLOCK_SIZE=8 -DIMAGE_WIDTH={image_width}" CXXFLAGS="-DIMAGE_WIDTH={image_width}" cmake -B build --fresh . && cmake --build build'
hyperfine -w 3 -r 5 -L image_width 800,1200,1280,1920,2560,3840 --export-json=image_width.json --sort=command --shell=bash --setup "${SETUP}" "${CMD1}" "${CMD2}" "${CMD3}"
