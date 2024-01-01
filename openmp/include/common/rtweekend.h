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

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>

#include "def.h"

// Usings

using std::make_shared;
using std::shared_ptr;
using std::sqrt;

// Constants

const real_type infinity = std::numeric_limits<real_type>::infinity();
const real_type pi = 3.1415926535897932385;

// Utility Functions

inline real_type degrees_to_radians(real_type degrees)
{
    return degrees * pi / 180.0;
}

inline real_type clamp(real_type x, real_type min, real_type max)
{
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

inline real_type random_real()
{
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0);
}

inline real_type random_real_r(unsigned int& seed)
{
    // Returns a random real in [0,1).
    return rand_r(&seed) / (RAND_MAX + 1.0);
}

inline real_type random_real(real_type min, real_type max)
{
    // Returns a random real in [min,max).
    return min + (max - min) * random_real();
}

inline real_type random_real_r(real_type min, real_type max, unsigned int& seed)
{
    // Returns a random real in [min,max).
    return min + (max - min) * random_real_r(seed);
}

inline int random_int(int min, int max)
{
    // Returns a random integer in [min,max].
    return static_cast<int>(random_real(min, max + 1));
}

inline int random_int_r(int min, int max, unsigned int& seed)
{
    // Returns a random integer in [min,max].
    return static_cast<int>(random_real_r(min, max + 1, seed));
}

// Common Headers

#include "ray.h"
#include "vec3.h"

#endif
