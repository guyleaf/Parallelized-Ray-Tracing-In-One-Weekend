#---------------------------------------------------------------------------------------------------
# CMake Build Configuration for the Ray Tracing Weekend Series
#
# See README.md for guidance.
#---------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.23.0)

# If CMAKE_CXX_COMPILER is defined but CMAKE_CUDA_HOST_COMPILER is not defined
# To follow the same ABI, we should follow the same compiler as CMAKE_CXX_COMPILER to avoid inconsistent configuration
# Otherwise, we should not touch.
# NOTE: CMAKE_CUDA_HOST_COMPILER will set -ccbin
if(DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  message("CMAKE_CUDA_HOST_COMPILER is not defined.")
  message("Set CMAKE_CUDA_HOST_COMPILER to ${CMAKE_CUDA_HOST_COMPILER} as default ")
endif()

project (CUDARTWeekend
  VERSION 3.0.0
  LANGUAGES CXX CUDA
)

# Export compile_commands.json for vscode intellisense or others
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Let's ensure -std=c++xx instead of -std=g++xx
set(CMAKE_CXX_EXTENSIONS OFF)

# Require CUDA version >= 12.0
find_package(CUDAToolkit 12 REQUIRED)

# Source
set(
  headers
  "${PROJECT_SOURCE_DIR}/include/common"
  "${PROJECT_SOURCE_DIR}/include/InOneWeekend"
)
set(
  sources
  "${PROJECT_SOURCE_DIR}/src/main.cu"
)

# Executables
add_executable(CUDAInOneWeekend ${sources})
target_include_directories(CUDAInOneWeekend PRIVATE ${headers})

# Require CUDA/C++ 11
set_target_properties(
  CUDAInOneWeekend PROPERTIES
  CXX_STANDARD 11
  CUDA_STANDARD 11
  CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD_REQUIRED ON
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "all-major"
)

# Link CUDA library
target_link_libraries(CUDAInOneWeekend PRIVATE CUDA::cudart CUDA::curand)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  # In CMake, only append -g by default. We will need -G, too.
  target_compile_options(CUDAInOneWeekend PRIVATE -G)
endif()

# Enable all warnings and address sanitizer if in DEBUG mode
# NOTE: Currently, intellisense won't detect cuda library correctly due to adding compile options
# target_compile_options(CUDAInOneWeekend PUBLIC -Wall -Wextra -Wno-deprecated-declarations)
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#   target_compile_options(CUDAInOneWeekend PRIVATE -fsanitize=address -fdiagnostics-color=always -static-libasan)
#   target_link_options(CUDAInOneWeekend PRIVATE -fsanitize=address -fdiagnostics-color=always -static-libasan)
# endif()
