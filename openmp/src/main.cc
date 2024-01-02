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

#include <omp.h>

#include <iostream>
#include <vector>

#include "camera.h"
#include "color.h"
#include "def.h"
#include "hittable_list.h"
#include "material.h"
#include "rtweekend.h"
#include "sphere.h"

color ray_color(const ray& r, const hittable& world, int depth,
                unsigned int& seed)
{
    hit_record rec;

    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0) return color(0, 0, 0);

    if (world.hit(r, 0.001_r, infinity, rec, seed))
    {
        ray scattered;
        color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered, seed))
            return attenuation * ray_color(scattered, world, depth - 1, seed);
        return color(0, 0, 0);
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5_r * (unit_direction.y() + 1.0_r);
    return (1.0_r - t) * color(1.0_r, 1.0_r, 1.0_r) +
           t * color(0.5_r, 0.7_r, 1.0_r);
}

hittable_list random_scene()
{
    auto map_width = static_cast<int>(std::sqrt(MAP_SIZE));
    auto half_map_width = map_width / 2;
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5_r, 0.5_r, 0.5_r));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -half_map_width; a < half_map_width; a++)
    {
        for (int b = -half_map_width; b < half_map_width; b++)
        {
            auto choose_mat = random_real();
            point3 center(a + 0.9_r * random_real(), 0.2_r,
                          b + 0.9_r * random_real());

            if ((center - point3(4, 0.2_r, 0)).length() > 0.9_r)
            {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8_r)
                {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(
                        make_shared<sphere>(center, 0.2_r, sphere_material));
                }
                else if (choose_mat < 0.95_r)
                {
                    // metal
                    auto albedo = color::random(0.5_r, 1);
                    auto fuzz = random_real(0, 0.5_r);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(
                        make_shared<sphere>(center, 0.2_r, sphere_material));
                }
                else
                {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5_r);
                    world.add(
                        make_shared<sphere>(center, 0.2_r, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5_r);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0_r, material1));

    auto material2 = make_shared<lambertian>(color(0.4_r, 0.2_r, 0.1_r));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0_r, material2));

    auto material3 = make_shared<metal>(color(0.7_r, 0.6_r, 0.5_r), 0.0_r);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0_r, material3));

    return world;
}

int main()
{
    // Image

    unsigned int seed = 5222;
    const auto aspect_ratio = 16.0_r / 9.0_r;
    const int image_width = IMAGE_WIDTH;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = SAMPLES_PER_PIXEL;
    const int max_depth = MAX_DEPTH;

    // World

    srand(seed);
    auto world = random_scene();

    // Camera

    point3 lookfrom(13, 2, 3);
    point3 lookat(0, 0, 0);
    vec3 vup(0, 1, 0);
    auto dist_to_focus = 10.0_r;
    auto aperture = 0.1_r;

    camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture,
               dist_to_focus);

    // Render

    std::vector<color> image(image_width * image_height);
#pragma omp parallel for collapse(2) firstprivate(seed)
    for (int j = 0; j < image_height; j++)
    {
        for (int i = 0; i < image_width; i++)
        {
            color pixel_color(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s)
            {
                auto u = (i + random_real_r(seed)) / (image_width - 1);
                auto v = (j + random_real_r(seed)) / (image_height - 1);
                ray r = cam.get_ray_r(u, v, seed);
                pixel_color += ray_color(r, world, max_depth, seed);
            }

            int index = j * image_width + i;
            image[index] = pixel_color;
        }
    }

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int j = image_height - 1; j >= 0; j--)
    {
        for (int i = 0; i < image_width; i++)
        {
            int index = j * image_width + i;
            write_color(std::cout, image[index], samples_per_pixel);
        }
    }

    std::cerr << "\nDone.\n";
}
