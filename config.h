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

// All possible grid geometries
#define GRID_GEOMETRY_SQUARE   0 // square tiling, like conway's game of life
#define GRID_GEOMETRY_TRIANGLE 1 // triangular tiling
#define GRID_GEOMETRY_HEXAGON  2 // hexagonal tiling

// All possible cell neighbourhoods
#define CELL_NEIGHBOURHOOD_TYPE_VERTEX 0 // a cell is in the current cell's neighbourhood iff it shares a vertex
#define CELL_NEIGHBOURHOOD_TYPE_EDGE   1 // a cell is in the current cell's neighbourhood iff it shares an edge

// All possible ways to decide whether a cell is "fit", or not.
#define FITNESS_EVAL_STATE 0 // counts the number of cells with a given state compared to the overall number of cells
#define FITNESS_EVAL_UPDATE 1 // counts the number of cell updates compared to the overall number of cells

// All possible ways to compute the actual fitness from the number of "fit" cells
// See https://www.desmos.com/calculator/b8daos2dqt for visual explanation.
#define FITNESS_FN_TYPE_ABS 0
#define FITNESS_FN_TYPE_LINEAR 1
#define FITNESS_FN_TYPE_LIKELIHOOD 2

/**************************
 * START OF CONFIGURATION *
 **************************/

// # Grid Geometry

// [uint] simulation grid width
#define GRID_WIDTH  200
// [uint] simulation grid height
#define GRID_HEIGHT 100
// [enum] the shape of the grid's cells (square for Conway's GoL)
#define GRID_GEOMETRY GRID_GEOMETRY_TRIANGLE
// [enum] which cells are considered in the neighbourhood (vertex for Conway's GoL)
#define CELL_NEIGHBOURHOOD_TYPE CELL_NEIGHBOURHOOD_TYPE_VERTEX
// [uchar] number of states a cell can become (2 for Conway's GoL)
#define CELL_STATES 3

// Conway's GoL:
/* #define GRID_GEOMETRY GRID_GEOMETRY_SQUARE */
/* #define CELL_NEIGHBOURHOOD_TYPE CELL_NEIGHBOURHOOD_TYPE_VERTEX */
/* #define CELL_STATES 2 */

// # Seeking parameters (genetic algorithm)
#define POPULATION_SIZE 512

// ## Fitness function
// [uint] the index of the iteration to start cumulating the fitness error from
#define FITNESS_EVAL_FROM 100
// [uint] the number of iterations to cumulate the fitness error for
#define FITNESS_EVAL_LEN 10

// Proportion
// [enum] which cell evaluation method to use (decide whether a given cell is "fit", or not)
#define FITNESS_EVAL FITNESS_EVAL_STATE
// [uint] which state to count the proportion of
#define FITNESS_EVAL_STATE_INDEX 0
// [enum] which way to evaluate the fitness function
#define FITNESS_FN_TYPE FITNESS_FN_TYPE_ABS
// [float] what is the target value of the fitness function (the value at which there is a maximum)
#define FITNESS_FN_TARGET 0.9

// # Miscellaneous

// [uint] execution block width and height
#define BLOCK_LENGTH 16
// [bool] whether to verify the GPU simulation with an equivalent CPU simulation
#define CPU_VERIFY false
// [bool] `true` to keep aspect ratio, `false` to stretch to window
#define KEEP_ASPECT_RATIO true
// [bool] `true` if multisampling should be enabled to fix "jagged" edges, `false` otherwise
#define USE_MULTISAMPLING true
// [uint] number of samples per pixel to use when multisampling is enabled
#define MULTISAMPLING_SAMPLES 16
// [float/uint] number of iterations per second
#define FRAMERATE 2
// [uint] number of milliseconds to wait between iterations (overriden by FRAMERATE)
/* #define SLEEP_MS 5000 */
// [uint] initial window width
#define WINDOW_WIDTH 800
// [uint] initial window height
#define WINDOW_HEIGHT 800
// [uint] max number of frames exported as PNG
/* #define EXPORT_FRAMES 100 */
#define PROMPT_TO_START false
// Close the application after this many frames. Useful for profiling.
/* #define EXIT_AFTER_FRAMES 10 */
// Close the application after this many populations. Useful for profiling.
#define EXIT_AFTER_POPULATIONS 3
#define USE_SHARED_MEMORY true
#define DETERMINISTIC_RANDOMNESS false

/************************
 * END OF CONFIGURATION *
 ************************/


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
