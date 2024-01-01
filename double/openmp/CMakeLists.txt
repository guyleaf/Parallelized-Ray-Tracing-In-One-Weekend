#---------------------------------------------------------------------------------------------------
# CMake Build Configuration for the Ray Tracing Weekend Series
#
# See README.md for guidance.
#---------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.23.0)

project (DoubleOMPRTWeekend
  VERSION 3.0.0
  LANGUAGES CXX
)

# Export compile_commands.json for vscode intellisense or others
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Let's ensure -std=c++xx instead of -std=g++xx
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenMP library
find_package(OpenMP REQUIRED)

# Source
set(
  headers
  "${PROJECT_SOURCE_DIR}/include/common"
  "${PROJECT_SOURCE_DIR}/include/InOneWeekend"
)
set(
  sources
  "${PROJECT_SOURCE_DIR}/src/main.cc"
)

# Executables
add_executable(DoubleOMPInOneWeekend ${sources})
target_include_directories(DoubleOMPInOneWeekend PRIVATE ${headers})

# Require C++11
set_target_properties(
  DoubleOMPInOneWeekend PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED ON
)

# Link OpenMP library
target_link_libraries(DoubleOMPInOneWeekend PRIVATE OpenMP::OpenMP_CXX)
