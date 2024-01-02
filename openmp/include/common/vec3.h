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

#include <cmath>
#include <iostream>

#include "def.h"

using std::fabs;
using std::fmin;
using std::sqrt;

class vec3
{
   public:
    vec3() : e{0, 0, 0} {}
    vec3(real_type e0, real_type e1, real_type e2) : e{e0, e1, e2} {}

    real_type x() const { return e[0]; }
    real_type y() const { return e[1]; }
    real_type z() const { return e[2]; }

    vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    real_type operator[](int i) const { return e[i]; }
    real_type &operator[](int i) { return e[i]; }

    vec3 &operator+=(const vec3 &v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    vec3 &operator*=(const real_type t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    vec3 &operator/=(const real_type t) { return *this *= 1 / t; }

    real_type length() const { return sqrt(length_squared()); }

    real_type length_squared() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    bool near_zero() const
    {
        // Return true if the vector is close to zero in all dimensions.
        const auto s = 1e-8_r;
        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    }

    inline static vec3 random()
    {
        return vec3(random_real(), random_real(), random_real());
    }

    inline static vec3 random(real_type min, real_type max)
    {
        return vec3(random_real(min, max), random_real(min, max),
                    random_real(min, max));
    }

    inline static vec3 random_r(unsigned int &seed)
    {
        return vec3(random_real_r(seed), random_real_r(seed),
                    random_real_r(seed));
    }

    inline static vec3 random_r(real_type min, real_type max,
                                unsigned int &seed)
    {
        return vec3(random_real_r(min, max, seed),
                    random_real_r(min, max, seed),
                    random_real_r(min, max, seed));
    }

   public:
    real_type e[3];
};

// Type aliases for vec3
using point3 = vec3;  // 3D point
using color = vec3;   // RGB color

// vec3 Utility Functions

inline std::ostream &operator<<(std::ostream &out, const vec3 &v)
{
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

inline vec3 operator+(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline vec3 operator-(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline vec3 operator*(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline vec3 operator*(real_type t, const vec3 &v)
{
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

inline vec3 operator*(const vec3 &v, real_type t) { return t * v; }

inline vec3 operator/(vec3 v, real_type t) { return (1 / t) * v; }

inline real_type dot(const vec3 &u, const vec3 &v)
{
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

inline vec3 cross(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline vec3 unit_vector(vec3 v) { return v / v.length(); }

inline vec3 random_in_unit_disk()
{
    while (true)
    {
        auto p = vec3(random_real(-1, 1), random_real(-1, 1), 0);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

inline vec3 random_in_unit_disk_r(unsigned int &seed)
{
    while (true)
    {
        auto p =
            vec3(random_real_r(-1, 1, seed), random_real_r(-1, 1, seed), 0);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

inline vec3 random_in_unit_sphere()
{
    while (true)
    {
        auto p = vec3::random(-1, 1);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

inline vec3 random_in_unit_sphere_r(unsigned int &seed)
{
    while (true)
    {
        auto p = vec3::random_r(-1, 1, seed);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

inline vec3 random_unit_vector()
{
    return unit_vector(random_in_unit_sphere());
}

inline vec3 random_unit_vector_r(unsigned int &seed)
{
    return unit_vector(random_in_unit_sphere_r(seed));
}

inline vec3 random_in_hemisphere(const vec3 &normal)
{
    vec3 in_unit_sphere = random_in_unit_sphere();
    if (dot(in_unit_sphere, normal) >
        0)  // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

inline vec3 random_in_hemisphere_r(const vec3 &normal, unsigned int &seed)
{
    vec3 in_unit_sphere = random_in_unit_sphere_r(seed);
    if (dot(in_unit_sphere, normal) >
        0)  // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

inline vec3 reflect(const vec3 &v, const vec3 &n)
{
    return v - 2 * dot(v, n) * n;
}

inline vec3 refract(const vec3 &uv, const vec3 &n, real_type etai_over_etat)
{
    auto cos_theta = fmin(dot(-uv, n), 1.0_r);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

#endif
