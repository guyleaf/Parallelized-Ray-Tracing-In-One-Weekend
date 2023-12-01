#---------------------------------------------------------------------------------------------------
# CMake Build Configuration for the Ray Tracing Weekend Series
#
# See README.md for guidance.
#---------------------------------------------------------------------------------------------------

# Starting from version 3.8 (3.9 for MSVC),
# CUDA C++ becomes an intrinsically supported language.
cmake_minimum_required ( VERSION 3.9.0...3.27.0 )

project ( RTWeekend
  VERSION 3.0.0
  LANGUAGES CXX CUDA
)

include(CheckLanguage)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA Support enabled.")

    # NOTE: FindCUDA is deprecated after CMake 3.18
    include(FindCUDA)
    set(CUDA_ARCH_LIST Auto CACHE STRING  "List of CUDA architectures (e.g. Pascal, Volta, etc) or \
                                           compute capability versions (6.1, 7.0, etc) to generate code for. \
                                           Set to Auto for automatic detection (default).")
    cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
    list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})
else()
    message(STATUS "CUDA Support disabled.")
endif()

# Set to c++11
set ( CMAKE_CXX_STANDARD 11 )
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Source
set ( COMMON_ALL
  src/common/rtweekend.h
  src/common/camera.h
  src/common/ray.h
  src/common/vec3.h
)

set ( SOURCE_ONE_WEEKEND
  ${COMMON_ALL}
  src/InOneWeekend/hittable.h
  src/InOneWeekend/hittable_list.h
  src/InOneWeekend/material.h
  src/InOneWeekend/sphere.h
  src/InOneWeekend/main.cu
)

# Executables
add_executable(inOneWeekend      ${SOURCE_ONE_WEEKEND})

include_directories(src/common)
