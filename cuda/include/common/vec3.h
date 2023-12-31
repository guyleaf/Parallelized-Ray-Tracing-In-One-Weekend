#ifndef VEC3_H
#define VEC3_H
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright
// and related and neighboring rights to this software to the public domain
// worldwide. This software is distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public
// Domain Dedication along with this software. If not, see
// <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include <curand_kernel.h>

#include <cmath>
#include <iostream>

using std::fabs;
using std::sqrt;

// The vec3 class is designed for use on both the CPU and GPU, hence all member
// functions are annotated with __host__ __device__. This includes helpers and
// operator overloads directly involving vec3, with the exception of streaming
// operators (<<, >>), which are I/O operations rather than computations.
class vec3
{
   public:
    __host__ __device__ vec3() : e{0, 0, 0} {}
    __host__ __device__ vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

    __host__ __device__ double x() const { return e[0]; }
    __host__ __device__ double y() const { return e[1]; }
    __host__ __device__ double z() const { return e[2]; }

    __host__ __device__ vec3 operator-() const
    {
        return vec3(-e[0], -e[1], -e[2]);
    }
    __host__ __device__ double operator[](int i) const { return e[i]; }
    __host__ __device__ double& operator[](int i) { return e[i]; }

    __host__ __device__ vec3& operator+=(const vec3& v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ vec3& operator*=(const double t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ vec3& operator/=(const double t)
    {
        return *this *= 1 / t;
    }

    __host__ __device__ double length() const { return sqrt(length_squared()); }

    __host__ __device__ double length_squared() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    __host__ __device__ bool near_zero() const
    {
        // Return true if the vector is close to zero in all dimensions.
        const auto s = 1e-8;
        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    }

    inline static vec3 random()
    {
        return vec3(random_double(), random_double(), random_double());
    }

    __device__ inline static vec3 random(curandState* rand_state)
    {
        return vec3(random_double(rand_state), random_double(rand_state),
                    random_double(rand_state));
    }

    inline static vec3 random(double min, double max)
    {
        return vec3(random_double(min, max), random_double(min, max),
                    random_double(min, max));
    }

    __device__ inline static vec3 random(double min, double max,
                                         curandState* rand_state)
    {
        return vec3(random_double(min, max, rand_state),
                    random_double(min, max, rand_state),
                    random_double(min, max, rand_state));
    }

    //
    // The following functions are specialized to not use
    // `curand_uniform_double()` but instead use the `rand_num` parameter.
    // Particularly used in the `random_scene` function to have the scene be the
    // same on the GPU and CPU.
    //

    /// @param rand_nums
    /// @param rand_idx The first index of the random number to be used. It will
    /// be incremented by the number of the random numbers used.
    __device__ inline static vec3 random_s(int* rand_nums, int& rand_idx)
    {
        // XXX: Since we're using GCC as our host compiler, its order of
        // evaluation appears to be right-to-left. The order of evaluation is
        // unspecified in C++. Changing the host compiler to Clang or MSVC will
        // likely break this, as well as on other platforms.
        auto e2 = random_double_s(rand_nums[rand_idx++]);
        auto e1 = random_double_s(rand_nums[rand_idx++]);
        auto e0 = random_double_s(rand_nums[rand_idx++]);
        return vec3(e0, e1, e2);
    }

    /// @param min
    /// @param max
    /// @param rand_nums
    /// @param rand_idx The first index of the random number to be used. It will
    /// be incremented by the number of the random numbers used.
    __device__ inline static vec3 random_s(double min, double max,
                                           int* rand_nums, int& rand_idx)
    {
        // XXX: Since we're using GCC as our host compiler, its order of
        // evaluation appears to be right-to-left. The order of evaluation is
        // unspecified in C++. Changing the host compiler to Clang or MSVC will
        // likely break this, as well as on other platforms.
        auto e2 = random_double_s(min, max, rand_nums[rand_idx++]);
        auto e1 = random_double_s(min, max, rand_nums[rand_idx++]);
        auto e0 = random_double_s(min, max, rand_nums[rand_idx++]);
        return vec3(e0, e1, e2);
    }

   public:
    double e[3];
};

// Type aliases for vec3
using point3 = vec3;  // 3D point
using color = vec3;   // RGB color

// vec3 Utility Functions

inline std::ostream& operator<<(std::ostream& out, const vec3& v)
{
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v)
{
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v)
{
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v)
{
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(double t, const vec3& v)
{
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, double t)
{
    return t * v;
}

__host__ __device__ inline vec3 operator/(vec3 v, double t)
{
    return (1 / t) * v;
}

__host__ __device__ inline double dot(const vec3& u, const vec3& v)
{
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v)
{
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v) { return v / v.length(); }

__device__ inline vec3 random_in_unit_disk(curandState* rand_state)
{
    while (true)
    {
        auto p = vec3(random_double(-1, 1, rand_state),
                      random_double(-1, 1, rand_state), 0);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

__device__ inline vec3 random_in_unit_sphere(curandState* rand_state)
{
    while (true)
    {
        auto p = vec3::random(-1, 1, rand_state);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

__device__ inline vec3 random_unit_vector(curandState* rand_state)
{
    return unit_vector(random_in_unit_sphere(rand_state));
}

__device__ inline vec3 random_in_hemisphere(const vec3& normal,
                                            curandState* rand_state)
{
    vec3 in_unit_sphere = random_in_unit_sphere(rand_state);
    if (dot(in_unit_sphere, normal) >
        0.0)  // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

__host__ __device__ inline vec3 reflect(const vec3& v, const vec3& n)
{
    return v - 2 * dot(v, n) * n;
}

__host__ __device__ inline vec3 refract(const vec3& uv, const vec3& n,
                                        double etai_over_etat)
{
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

#endif
