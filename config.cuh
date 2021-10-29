#pragma once

// All possible grid geometries
#define GRID_GEOMETRY_SQUARE   0 // square tiling, like conway's game of life
#define GRID_GEOMETRY_TRIANGLE 1 // triangular tiling
#define GRID_GEOMETRY_HEXAGON  2 // hexagonal tiling

// All possible cell neighbourhoods
#define CELL_NEIGHBOURHOOD_TYPE_VERTEX 0 // a cell is in the current cell's neighbourhood iff it shares a vertex
#define CELL_NEIGHBOURHOOD_TYPE_EDGE   1 // a cell is in the current cell's neighbourhood iff it shares an edge


/**************************
 * START OF CONFIGURATION *
 **************************/

// [uint] simulation grid width
#define GRID_WIDTH  1000
// [uint] simulation grid height
#define GRID_HEIGHT 1000
// [enum] the shape of the grid's cells (square for Conway's GoL)
/* #define GRID_GEOMETRY GRID_GEOMETRY_TRIANGLE */
// [enum] which cells are considered in the neighbourhood (vertex for Conway's GoL)
/* #define CELL_NEIGHBOURHOOD_TYPE CELL_NEIGHBOURHOOD_TYPE_VERTEX */
// [uchar] number of states a cell can become (2 for Conway's GoL)
/* #define CELL_STATES 8 */

// Conway's GoL:
#define GRID_GEOMETRY GRID_GEOMETRY_SQUARE
#define CELL_NEIGHBOURHOOD_TYPE CELL_NEIGHBOURHOOD_TYPE_VERTEX
#define CELL_STATES 2

// [uint] execution block width and height
#define BLOCK_LENGTH 16
// [bool] whether to verify the GPU simulation with an equivalent CPU simulation
#define CPU_VERIFY IS_DEBUG

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
        #define CELL_NEIGHBOURHOOD_SIZE 4
    #endif
#elif GRID_GEOMETRY == GRID_GEOMETRY_HEXAGON
    #if CELL_NEIGHBOURHOOD_TYPE == CELL_NEIGHBOURHOOD_TYPE_VERTEX
        #define CELL_NEIGHBOURHOOD_SIZE 6
    #elif CELL_NEIGHBOURHOOD_TYPE == CELL_NEIGHBOURHOOD_TYPE_EDGE
        #define CELL_NEIGHBOURHOOD_SIZE 6
    #endif
#endif

#define BLOCK_AREA            (BLOCK_LENGTH * BLOCK_LENGTH)                     // threads per execution block
#define GRID_WIDTH_IN_BLOCKS  ((GRID_WIDTH + BLOCK_LENGTH - 1) / BLOCK_LENGTH)  // execution grid width
#define GRID_HEIGHT_IN_BLOCKS ((GRID_HEIGHT + BLOCK_LENGTH - 1) / BLOCK_LENGTH) // execution grid height
