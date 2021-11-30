#pragma once

// Utils
#define POW2_CEIL(x_) ({ \
    unsigned int x = x_; \
    x -= 1;              \
    x = x | (x >> 1);    \
    x = x | (x >> 2);    \
    x = x | (x >> 4);    \
    x = x | (x >> 8);    \
    x = x | (x >>16);    \
    x + 1;               \
})
#define GET_POW2_CEIL(field, x_)  \
    field = x_;                   \
    field -= 1;                   \
    field = field | (field >> 1); \
    field = field | (field >> 2); \
    field = field | (field >> 4); \
    field = field | (field >> 8); \
    field = field | (field >>16); \
    field = field + 1;

// Automatically derived constants from the configuration
#if GRID_GEOMETRY == GRID_GEOMETRY_SQUARE
    #if CELL_NEIGHBOURHOOD_TYPE == CELL_NEIGHBOURHOOD_TYPE_VERTEX
        #define CELL_NEIGHBOURHOOD_SIZE 8
    #elif CELL_NEIGHBOURHOOD_TYPE == CELL_NEIGHBOURHOOD_TYPE_EDGE
        #define CELL_NEIGHBOURHOOD_SIZE 4
    #endif
#elif GRID_GEOMETRY == GRID_GEOMETRY_TRIANGLE
    #if CELL_NEIGHBOURHOOD_TYPE == CELL_NEIGHBOURHOOD_TYPE_VERTEX
        #define CELL_NEIGHBOURHOOD_SIZE 12
    #elif CELL_NEIGHBOURHOOD_TYPE == CELL_NEIGHBOURHOOD_TYPE_EDGE
        #define CELL_NEIGHBOURHOOD_SIZE 3
    #endif
#elif GRID_GEOMETRY == GRID_GEOMETRY_HEXAGON
    #if CELL_NEIGHBOURHOOD_TYPE == CELL_NEIGHBOURHOOD_TYPE_VERTEX
        #define CELL_NEIGHBOURHOOD_SIZE 6
    #elif CELL_NEIGHBOURHOOD_TYPE == CELL_NEIGHBOURHOOD_TYPE_EDGE
        #define CELL_NEIGHBOURHOOD_SIZE 6
    #endif
#endif

#if GRID_GEOMETRY == GRID_GEOMETRY_SQUARE
    #define CELL_VERTICES 4
#elif GRID_GEOMETRY == GRID_GEOMETRY_TRIANGLE
    #define CELL_VERTICES 3
#elif GRID_GEOMETRY == GRID_GEOMETRY_HEXAGON
    #define CELL_VERTICES 6
#endif

#define GRID_AREA (GRID_WIDTH * GRID_HEIGHT)
#define BLOCK_AREA            (BLOCK_LENGTH * BLOCK_LENGTH)                     // threads per execution block
#define GRID_WIDTH_IN_BLOCKS  ((GRID_WIDTH + BLOCK_LENGTH - 1) / BLOCK_LENGTH)  // execution grid width
#define GRID_HEIGHT_IN_BLOCKS ((GRID_HEIGHT + BLOCK_LENGTH - 1) / BLOCK_LENGTH) // execution grid height
#define GRID_AREA_IN_BLOCKS (GRID_WIDTH_IN_BLOCKS * GRID_HEIGHT_IN_BLOCKS)
#define GRID_PITCH (POW2_CEIL(GRID_WIDTH))
#define GRID_DIM_BLOCKS (dim3(GRID_WIDTH_IN_BLOCKS, GRID_HEIGHT_IN_BLOCKS))
#define GRID_DIM_THREADS (dim3(BLOCK_LENGTH, BLOCK_LENGTH))
#define GET_GRID_PITCH(field) GET_POW2_CEIL(field, GRID_WIDTH)
#define GRID_AREA_WITH_PITCH (GRID_PITCH * GRID_HEIGHT)
#define SHARED_SUBGRID_MARGIN 2 // a 2 cell margin on each side (max of 1 for square, 1 for hex, 2 for triangle; also simplifies neighbour cell addressing)
#define SHARED_SUBGRID_LENGTH (BLOCK_LENGTH + 2 * SHARED_SUBGRID_MARGIN)
#define SHARED_SUBGRID_AREA (SHARED_SUBGRID_LENGTH * SHARED_SUBGRID_LENGTH) // number of states loaded to shared memory
#define SHARED_SUBGRID_LOAD_ITERATIONS ((SHARED_SUBGRID_AREA + BLOCK_AREA - 1) / BLOCK_AREA) // number of iterations required to load the shared subgrid in parallel using all threads in the block

#define FITNESS_EVAL_TO (FITNESS_EVAL_FROM + FITNESS_EVAL_LEN)

#ifdef FRAMERATE
    #ifndef SLEEP_MS
        #define SLEEP_MS ((u64) (1000.0 / FRAMERATE))
    #endif
#endif
