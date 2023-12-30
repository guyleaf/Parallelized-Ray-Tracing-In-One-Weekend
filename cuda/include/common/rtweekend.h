#ifndef RTWEEKEND_H
#define RTWEEKEND_H
//==============================================================================================
// To the extent possible under law, the author(s) have dedicated all copyright
// and related and neighboring rights to this software to the public domain
// worldwide. This software is distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public
// Domain Dedication along with this software. If not, see
// <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>

// Usings

using std::make_shared;
using std::shared_ptr;
using std::sqrt;

// Constants

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions

__host__ __device__ inline double degrees_to_radians(double degrees)
{
    return degrees * pi / 180.0;
}

__host__ __device__ inline double clamp(double x, double min, double max)
{
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

inline double random_double()
{
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0);
}

// Returns a random real in [0,1).
__device__ inline double random_double(curandState* rand_state)
{
    return curand_uniform_double(rand_state);
}

inline double random_double(double min, double max)
{
    // Returns a random real in [min,max).
    return min + (max - min) * random_double();
}

// Returns a random real in [min,max).
__device__ inline double random_double(double min, double max,
                                       curandState* rand_state)
{
    return min + (max - min) * random_double(rand_state);
}

inline int random_int(int min, int max)
{
    // Returns a random integer in [min,max].
    return static_cast<int>(random_double(min, max + 1));
}

// Returns a random integer in [min,max].
__device__ inline int random_int(int min, int max, curandState* rand_state)
{
    return static_cast<int>(random_double(min, max + 1, rand_state));
}

// Common Headers

#include "ray.h"
#include "vec3.h"

#endif
