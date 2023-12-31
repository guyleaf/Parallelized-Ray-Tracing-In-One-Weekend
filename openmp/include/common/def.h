#ifndef DEF_H

#ifndef IMAGE_WIDTH
#define IMAGE_WIDTH 1200
#endif
#ifndef SAMPLES_PER_PIXEL
#define SAMPLES_PER_PIXEL 10
#endif
#ifndef MAX_DEPTH
#define MAX_DEPTH 50
#endif
#ifndef MAP_SIZE
#define MAP_SIZE 22
#endif

#ifdef USE_FLOAT
using real_type = float;
#else
using real_type = double;
#endif

#endif
