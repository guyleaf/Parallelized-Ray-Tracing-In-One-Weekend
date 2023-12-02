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

Next, we are actually going to launch the function on the GPU. There are three types of [Function Execution Space Specifiers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-execution-space-specifiers), specifying whether a function should be called by and executed on the host (CPU) or device (GPU).

The function `render` is to be executed on the GPU but called from the CPU, so a `__global__` specifier is required. All other functions called by `render` are then called by and executed on GPU, so we add a `__device__` specifier. Some functions are called by `main` to execute on the CPU but can also be called on the GPU. In those cases, both `__host__` and `__device__` specifiers are used. Functions being called by these functions have this property propagate, thus also needing both specifiers.

While adding the function execution space specifiers, we encountered three types of functions that couldn't be directly converted into GPU code using specifiers due to their reliance on functions implemented by the standard library. These functions are the `random_xxx` function, `std::shared_ptr`, and `std::vector`.

## Random Number Generation

Fortunately, CUDA does support random number generation on the GPU through [its API](https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-overview).

We utilize `curand` as a replacement for `rand`, which requires a `curandState` to store the state. Each thread holds its own state. All functions that call the random generation functions are modified to take this additional parameter.

## Smart Pointer

A smart pointer handles memory resources without the need for intervention by programmers. While it is not designed to be used on the GPU, and CUDA doesn't have its own implementation at this time. So, we have to use raw pointers and remember to release the memory resources explicitly.

## Standard Container

Standard containers cannot be used in kernel function because their functions are not specified with `__device__`. We'll have to use C-style arrays, particularly for storing the _hittables_.

After these modifications, the code can again be compiled successfully.
However, when we run the program, it completes vastly with all pixels in black (0, 0, 0).
This is because we didn't allocate any memory on the device; all memories are on the CPU.
To address this, we decide where to allocate the memory based on their usage:
- The pixel buffer is computed by the GPU and written out by the CPU: allocate on Unified Memory with `cudaMallocManaged`.
- The random states are used only in kernel functions: allocate on the GPU with `cudaMalloc`.
- The hittables in the world are referenced only in kernel functions: allocate on the GPU.

Here we encountered a problem: the world is randomly initialized on the CPU. If we are to initialize it on the CPU, the hittables have to be allocated on unified memory so that both CPU and GPU can share it. To do so, we have to allocate them with `cudaMallocManaged` instead of `new`, and also in the implementation of related classes, i.e., we cannot use `delete` in their destructors, which doesn't seem sound.

Notably, the operators `new` and `delete` can be used on both CPU and GPU, but they have different meanings. A `new` on the CPU has the memory allocated on the CPU, and the corresponding `delete` can only be called on the CPU too. The same is true on the GPU.
Knowing this, we have another solution: initialize the world on the GPU, too. We then prepare another random state to use in the initialization of the world, and everything else should then work smoothly.
