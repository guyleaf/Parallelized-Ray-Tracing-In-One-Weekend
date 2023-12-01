#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H
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

#include "hittable.h"

#include <memory>
#include <vector>


class hittable_list : public hittable  {
    public:
        hittable_list() {}
        // The ownership of the object is transferred.
        hittable_list(hittable* object) { add(object); }

        void clear() { objects.clear(); }
        // The ownership of the object is transferred.
        void add(hittable* object) { objects.push_back(object); }

        __device__ virtual bool hit(
            const ray& r, double t_min, double t_max, hit_record& rec) const override;

        ~hittable_list() {
            for (auto* hittable : objects) {
                delete hittable;
            }
        }

    public:
        std::vector<hittable*> objects;
};


__device__ bool hittable_list::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
    hit_record temp_rec;
    auto hit_anything = false;
    auto closest_so_far = t_max;

    for (const auto& object : objects) {
        if (object->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}


#endif
