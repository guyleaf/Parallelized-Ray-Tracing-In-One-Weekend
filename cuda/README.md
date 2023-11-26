# Parallelized Ray Tracing In One Weekend With CUDA

> [!note]
> :construction: This parallel implementation is under active development.

## About

This sub-project is dedicated to parallelizing the renowned ray tracing tutorial using  _[CUDA](https://developer.nvidia.com/cuda-zone)_, enabling efficient execution on NVIDIA GPUs. The focus is on parallelizing the initial book of the _Ray Tracing in One Weekend Book Series_, titled _In One Week_. The parallelization effort for other books in the series is designated for future work.

> [!note]
> The source code used in this sub-project corresponds to version 3.2.3.

## Prerequisites

- CUDA-capable GPU
- A supported version of Linux with a gcc compiler and toolchain
- CUDA Toolkit (available at https://developer.nvidia.com/cuda-downloads)

## Building and Running

To build, go to the root of the sub-project directory and run the following commands to create the _debug_ version of the `inOneWeekend` executable:

```sh
cmake -B build
cmake --build build
```

For the _release_ version, run the following commands instead:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

On Windows, you can build either _debug_ (the default) or _release_ (the optimized version). To specify this, use the `--config <debug|release>` option.

### Running The Programs

On Linux or OSX, from the terminal, run like this:

```sh
build/inOneWeekend > image.ppm
```

On Windows, run like this:

```sh
build\debug\inOneWeekend > image.ppm
```

or, run the optimized version (if you've built with `--config release`):

```sh
build\release\inOneWeekend > image.ppm
```
