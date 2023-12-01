#ifndef HITTABLE_H
#define HITTABLE_H
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "rtweekend.h"

class material;


struct hit_record {
    point3 p;
    vec3 normal;
    // The record doesn't own the pointer.
    // The user is responsible to ensure the lifetime of the material is longer than the record.
    material* mat_ptr;
    double t;
    bool front_face;

    __device__ inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal :-outward_normal;
    }
};


class hittable {
    public:
        __device__ virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const = 0;

        virtual ~hittable() = default;

        // See C.67: A polymorphic class should suppress public copy/move in C++ Core Guidelines
        // https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c67-a-polymorphic-class-should-suppress-public-copymove

        hittable(const hittable&) = delete;
        hittable(hittable&&) = delete;

        // User-declared copy constructor prevents compiler from generating default constructor.
        // Add it explicitly.
        hittable() = default;
};


#endif
