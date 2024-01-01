#!/usr/bin/env bash

CPUS="0-9,20-29"
# CPUS="10-19,30-39"
hyperfine -w 3 -r 5 --export-json=serial.json "taskset -c ${CPUS} ./build/serial/InOneWeekend"
