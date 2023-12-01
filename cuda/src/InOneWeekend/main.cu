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

#include "camera.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"

#include <iostream>
#include <curand_kernel.h>


__device__ color ray_color(const ray& r, const hittable& world, int depth, curandState* rand_state) {
    hit_record rec;

    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0)
        return color(0,0,0);

    if (world.hit(r, 0.001, infinity, rec)) {
        ray scattered;
        color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered, rand_state))
            return attenuation * ray_color(scattered, world, depth-1, rand_state);
        return color(0,0,0);
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

// The scene is set up by on the GPU.
hittable_list random_scene() {
    hittable_list world;

    auto ground_material = new lambertian(color(0.5, 0.5, 0.5));
    world.add(new sphere(point3(0,-1000,0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                material* sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = new lambertian(albedo);
                    world.add(new sphere(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = new metal(albedo, fuzz);
                    world.add(new sphere(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = new dielectric(1.5);
                    world.add(new sphere(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = new dielectric(1.5);
    world.add(new sphere(point3(0, 1, 0), 1.0, material1));

    auto material2 = new lambertian(color(0.4, 0.2, 0.1));
    world.add(new sphere(point3(-4, 1, 0), 1.0, material2));

    auto material3 = new metal(color(0.7, 0.6, 0.5), 0.0);
    world.add(new sphere(point3(4, 1, 0), 1.0, material3));

    return world;
}

__global__ void render(vec3* buffer, int image_width, int image_height,
            hittable_list world, camera cam, int max_depth,
            int samples_per_pixel, curandState* rand_states) {
    const auto j = threadIdx.y + blockDim.y * blockIdx.y;
    const auto i = threadIdx.x + blockDim.x * blockIdx.x;
    const auto pixel_idx = j * image_width + i;
    color pixel_color(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; ++s) {
        auto u = (i + random_double(&rand_states[pixel_idx])) / (image_width - 1);
        auto v = (j + random_double(&rand_states[pixel_idx])) / (image_height - 1);
        ray r = cam.get_ray(u, v, &rand_states[pixel_idx]);
            pixel_color += ray_color(r, world, max_depth, &rand_states[pixel_idx]);
        }
        pixel_color /= samples_per_pixel;
        pixel_color[0] = std::sqrt(pixel_color[0]);
        pixel_color[1] = std::sqrt(pixel_color[1]);
        pixel_color[2] = std::sqrt(pixel_color[2]);
        buffer[pixel_idx] = pixel_color;
}

__global__ void init_curand_state(curandState* rand_states) {
    const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(1234, idx, 0, &rand_states[idx]);
}

int main() {

    // Image

    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 1200;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 10;
    const int max_depth = 50;

    // World

    auto world = random_scene();

    // Camera

    point3 lookfrom(13,2,3);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;

    camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

    // Divide the workload

    auto thread_size = dim3(16, 16);
    auto block_size = dim3(image_width / thread_size.x, image_height / thread_size.y);

    // Prepare randome number generator to be used in the kernel function

    auto rand_states = new curandState[image_width * image_height];
    init_curand_state<<<thread_size, block_size>>>(rand_states);

    // Render

    vec3* buffer = new vec3[image_width * image_height];
    render<<<thread_size, block_size>>>(buffer, image_width, image_height, world, cam, max_depth, samples_per_pixel, rand_states);

    // Write color

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int j = image_height-1; j >= 0; --j) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
            const auto& pixel_color = buffer[j * image_width + i];
            std::cout << static_cast<int>(256 * clamp(pixel_color.x(), 0.0, 0.999)) << ' '
                      << static_cast<int>(256 * clamp(pixel_color.y(), 0.0, 0.999)) << ' '
                      << static_cast<int>(256 * clamp(pixel_color.z(), 0.0, 0.999)) << '\n';
        }
    }
    std::cerr << "\nDone.\n";

    delete[] rand_states;
    delete[] buffer;
}
