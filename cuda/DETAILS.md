The computation hotspot comes from the following nested loop:
```cpp
std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
for (int j = image_height-1; j >= 0; --j) {
    std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
    for (int i = 0; i < image_width; ++i) {
        color pixel_color(0,0,0);
        for (int s = 0; s < samples_per_pixel; ++s) {
            auto u = (i + random_double()) / (image_width-1);
            auto v = (j + random_double()) / (image_height-1);
            ray r = cam.get_ray(u, v);
/* => */    pixel_color += ray_color(r, world, max_depth);
        }
        write_color(std::cout, pixel_color, samples_per_pixel);
    }
}
```

The arrow `/* => */` points to the critical line.
We can have each pixel be computed by a thread.
Notice that we cannot write to `std::cout` from the GPU. Resolve this issue by allocating a chunk of memory with the size exactly the size of the image (width * size) on the device. After the GPU concludes its work, we copy the memory back to the host.

The first step is to separate CPU and GPU codes into different functions:
```cpp
vec3* buffer = new vec3[image_width * image_width];
render(buffer, image_width, image_height, world, cam, max_depth, samples_per_pixel);

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

delete[] buffer;
```

The codes that are to be executed on the GPU are extracted to the function `render`:
```cpp
void render(vec3* buffer, int image_width, int image_height,
            hittable_list world, camera cam, int max_depth,
            int samples_per_pixel) {
    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            color pixel_color(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s) {
                auto u = (i + random_double()) / (image_width - 1);
                auto v = (j + random_double()) / (image_height - 1);
                ray r = cam.get_ray(u, v);
                    pixel_color += ray_color(r, world, max_depth);
                }
                pixel_color /= samples_per_pixel;
                pixel_color[0] = std::sqrt(pixel_color[0]);
                pixel_color[1] = std::sqrt(pixel_color[1]);
                pixel_color[2] = std::sqrt(pixel_color[2]);
                buffer[j * image_width + i] = pixel_color;
        }
    }
}
```
